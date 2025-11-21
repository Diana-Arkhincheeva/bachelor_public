import logging
import random
import json
from dataclasses import dataclass
from typing import List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor

from privacyschedulingtools.total_weighted_completion_time.pup.solver.pup_parallel import (
    PupParallel,
)
from privacyschedulingtools.total_weighted_completion_time.entity.scheduling_parameters import (
    ParallelSchedulingParameters,
)
from privacyschedulingtools.total_weighted_completion_time.entity.parallel_schedule import (
    ParallelScheduleFactory,
    ParallelSchedule,
    ScheduledJob,
)
from privacyschedulingtools.total_weighted_completion_time.entity.job import Job
from privacyschedulingtools.total_weighted_completion_time.entity.adversary.parallel_adversary import (
    ParallelLocalWSPTAdversary,
)
from privacyschedulingtools.total_weighted_completion_time.entity.domain import (
    IntegerDomain,
)
from privacyschedulingtools.total_weighted_completion_time.pup.utility_functions.parallel_machine_utilities import (
    calculate_twct,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
    logger.addHandler(logging.FileHandler("gen_bad_epsilon_50_3.log"))


def load_schedules_from_json(
    file_path: str,
) -> List[Tuple[ParallelSchedule, ParallelSchedulingParameters]]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = [data]

    schedules = []
    for schedule_data in data:
        machines_data = schedule_data["machines"]
        job_map: Dict[int, Job] = {}

        for machine in machines_data:
            for ji in machine["jobs"]:
                jid = ji["job_id"]
                if jid not in job_map:
                    job_map[jid] = Job(jid, ji["processing_time"], ji["weight"])

        jobs = list(job_map.values())
        processing_times = [j.processing_time for j in jobs]
        weights = [j.weight for j in jobs]

        params = ParallelSchedulingParameters(
            job_count=len(jobs),
            machine_count=len(machines_data),
            processing_time_domain=IntegerDomain(
                min(processing_times), max(processing_times)
            ),
            weight_domain=IntegerDomain(min(weights), max(weights)),
        )

        allocation = []
        for machine in machines_data:
            m_sched = []
            for ji in machine["jobs"]:
                jid = ji["job_id"]
                st = ji.get("start_time", 0)
                m_sched.append(ScheduledJob(jid, st))
            allocation.append(m_sched)

        schedule = ParallelSchedule(job_map, allocation, params)
        schedules.append((schedule, params))

    return schedules


def allocation_to_job_tuples(allocation, jobs_map):
    out = []
    for machine in allocation:
        out.append(
            [
                (
                    jobs_map[sj.id].id,
                    jobs_map[sj.id].processing_time,
                    jobs_map[sj.id].weight,
                )
                for sj in machine
            ]
        )
    return out


def repair_and_fill(allocation_tuples, jobs_map):
    seen = set()
    for machine in allocation_tuples:
        new = []
        for t in machine:
            if t[0] not in seen:
                new.append(t)
                seen.add(t[0])
        machine[:] = new

    all_ids = set(jobs_map.keys())
    missing = list(all_ids - seen)
    for mid in missing:
        job = jobs_map[mid]
        loads = [sum(x[1] for x in m) for m in allocation_tuples]
        min_idx = min(range(len(loads)), key=lambda i: loads[i])
        allocation_tuples[min_idx].append((job.id, job.processing_time, job.weight))
    return allocation_tuples


@dataclass
class GeneticOptimizer:
    epsilon_attack_threshold: float
    params: ParallelSchedulingParameters
    n_attacks: int = 10
    twct_tolerance: float = 0.0
    cpu_workers: int = 4

    def calculate_median_epsilon_and_twct(
        self, schedule: ParallelSchedule
    ) -> Tuple[float, float]:
        adversary = ParallelLocalWSPTAdversary(self.params)
        epsilons = []
        for _ in range(self.n_attacks):
            attack_result = adversary.execute_attack(schedule)
            solver = PupParallel(
                schedule=schedule,
                privacy_threshold=self.epsilon_attack_threshold,
                utility_threshold=1.0,
                schedule_parameters=self.params,
            )
            privacy_loss, _ = solver.calculate_privacy_loss(schedule, attack_result)
            eps = max(privacy_loss) if privacy_loss else float("inf")
            epsilons.append(eps)
        median_eps = sorted(epsilons)[len(epsilons) // 2]
        twct = calculate_twct(schedule)
        return float(median_eps), float(twct)

    def crossover(
        self, parent1: ParallelSchedule, parent2: ParallelSchedule
    ) -> ParallelSchedule:
        jobs_map = parent1.jobs
        mcount = self.params.machine_count
        all_jobs = list(jobs_map.values())
        allocation_tuples = [[] for _ in range(mcount)]

        for j in all_jobs:
            # случайный выбор родителя,затем берем машину, где этот job находится в выбранном родителе
            src_parent = parent1 if random.random() < 0.5 else parent2
            found_machine = None
            for mid, m in enumerate(src_parent.allocation):
                if any(sj.id == j.id for sj in m):
                    found_machine = mid
                    break
            if found_machine is None:
                found_machine = random.randrange(mcount)
            allocation_tuples[found_machine].append((j.id, j.processing_time, j.weight))

        allocation_tuples = repair_and_fill(allocation_tuples, jobs_map)
        factory = ParallelScheduleFactory(self.params)
        return factory.from_job_tuple_list(allocation_tuples, schedule_type="wspt")

    def mutate(
        self, schedule: ParallelSchedule, mutation_rate: float = 0.25
    ) -> ParallelSchedule:
        factory = ParallelScheduleFactory(self.params)
        allocation_tuples = allocation_to_job_tuples(schedule.allocation, schedule.jobs)

        # несколько случайных перемещений задач между машинами
        ops = max(1, int(len(allocation_tuples) * 3))
        for _ in range(ops):
            if random.random() < mutation_rate:
                src = random.randrange(len(allocation_tuples))
                if allocation_tuples[src]:
                    job_idx = random.randrange(len(allocation_tuples[src]))
                    job = allocation_tuples[src].pop(job_idx)
                    dst = random.randrange(len(allocation_tuples))
                    insert_pos = random.randint(0, len(allocation_tuples[dst]))
                    allocation_tuples[dst].insert(insert_pos, job)
        for mid in range(len(allocation_tuples)):
            if random.random() < mutation_rate and len(allocation_tuples[mid]) > 1:
                i, j = random.sample(range(len(allocation_tuples[mid])), 2)
                allocation_tuples[mid][i], allocation_tuples[mid][j] = (
                    allocation_tuples[mid][j],
                    allocation_tuples[mid][i],
                )

        allocation_tuples = repair_and_fill(allocation_tuples, schedule.jobs)
        return factory.from_job_tuple_list(allocation_tuples, schedule_type="wspt")

    def genetic_algorithm(
        self,
        initial_schedule: ParallelSchedule,
        population_size: int = 24,
        generations: int = 40,
        mutation_rate: float = 0.15,
    ) -> Tuple[float, float, ParallelSchedule]:
        baseline_twct = calculate_twct(initial_schedule)
        twct_limit = baseline_twct + self.twct_tolerance
        logger.info(
            f"Baseline TWCT = {baseline_twct:.2f}, TWCT limit = {twct_limit:.2f}"
        )

        population: List[ParallelSchedule] = [
            self.mutate(initial_schedule, mutation_rate=0.3)
            for _ in range(population_size)
        ]
        population[0] = initial_schedule

        def eval_candidate(
            sch: ParallelSchedule,
        ) -> Tuple[Tuple[float, float], ParallelSchedule]:
            med_eps, tw = self.calculate_median_epsilon_and_twct(sch)
            return (med_eps, tw), sch

        with ThreadPoolExecutor(max_workers=self.cpu_workers) as exe:
            futures = list(exe.map(eval_candidate, population))

        evaluated = list(futures)
        evaluated.sort(key=lambda x: (-x[0][0], x[0][1]))

        feasible = [e for e in evaluated if e[0][1] <= twct_limit]
        if feasible:
            best_eps, best_tw = feasible[0][0]
            best_schedule = feasible[0][1]
        else:
            evaluated.sort(key=lambda x: x[0][1])
            best_eps, best_tw = evaluated[0][0]
            best_schedule = evaluated[0][1]

        logger.info(f"Initial best: median ε={best_eps:.4f}, TWCT={best_tw:.2f}")

        for g in range(1, generations + 1):
            evaluated.sort(key=lambda x: (-x[0][0], x[0][1]))
            survivors = [s for (_, s) in evaluated[: max(2, population_size // 2)]]

            children = []
            while len(children) + len(survivors) < population_size:
                p1, p2 = random.sample(survivors, 2)
                child = self.crossover(p1, p2)
                child = self.mutate(child, mutation_rate)
                children.append(child)

            population = survivors + children

            evaluated = list(
                ThreadPoolExecutor(max_workers=self.cpu_workers).map(
                    eval_candidate, population
                )
            )

            evaluated.sort(key=lambda x: (-x[0][0], x[0][1]))

            feasible = [e for e in evaluated if e[0][1] <= twct_limit]
            if feasible:
                cand_eps, cand_tw = feasible[0][0]
                cand_sched = feasible[0][1]
                if (cand_eps > best_eps) or (
                    cand_eps == best_eps and cand_tw < best_tw
                ):
                    best_eps, best_tw, best_schedule = cand_eps, cand_tw, cand_sched
                    logger.info(
                        f"[Gen {g}] NEW BEST : median ε={best_eps:.4f}, TWCT={best_tw:.2f}"
                    )
            else:
                logger.debug(
                    f"[Gen {g}] Keine feasible Kandidaten (TWCT limit). Besten nach TWCT behalten."
                )

        logger.info(
            f"GA finished: best median ε={best_eps:.4f}, best TWCT={best_tw:.2f}"
        )
        return best_eps, best_tw, best_schedule


def run_genetic_for_schedule(args):
    schedule, params, idx = args
    optimizer = GeneticOptimizer(
        epsilon_attack_threshold=0.7,
        params=params,
        n_attacks=10,
        twct_tolerance=0.0,
        cpu_workers=4,
    )
    logger.info(f"GA started for schedule {idx}")
    best_eps, best_twct, best_schedule = optimizer.genetic_algorithm(
        initial_schedule=schedule,
        population_size=24,
        generations=30,
        mutation_rate=0.12,
    )
    logger.info(f"GA finished for {idx}: median ε={best_eps:.4f}, TWCT={best_twct:.2f}")

    res = {
        "index": idx,
        "best_median_epsilon": round(best_eps, 4),
        "best_twct": round(best_twct, 2),
        "machines": [],
    }
    for mid, machine in enumerate(best_schedule.allocation):
        machine_jobs = []
        for sj in machine:
            job = best_schedule.jobs[sj.id]
            machine_jobs.append(
                {
                    "job_id": job.id,
                    "start_time": sj.start_time,
                    "processing_time": job.processing_time,
                    "weight": job.weight,
                }
            )
        res["machines"].append({"machine_id": mid, "jobs": machine_jobs})
    return res


def main():
    input_file = "bad_epsilon_50_3.json"
    schedules = load_schedules_from_json(input_file)
    tasks = [(s, p, i) for i, (s, p) in enumerate(schedules)]

    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=40) as exe:
        results = list(exe.map(run_genetic_for_schedule, tasks))

    out_file = "gen_bad_epsilon_50_3.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"All GA results saved to {out_file}")


if __name__ == "__main__":
    main()
