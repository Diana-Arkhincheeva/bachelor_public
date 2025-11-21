import logging
import random
import json
from dataclasses import dataclass
from typing import List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

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
    logger.addHandler(logging.FileHandler("ga_30_3.log"))


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
    epsilon_threshold: float
    params: ParallelSchedulingParameters

    def calculate_metrics(self, schedule: ParallelSchedule) -> Tuple[float, float]:
        adversary = ParallelLocalWSPTAdversary(self.params)
        attack_result = adversary.execute_attack(schedule)
        solver = PupParallel(
            schedule,
            privacy_threshold=self.epsilon_threshold,
            utility_threshold=1.0,
            schedule_parameters=self.params,
        )
        privacy_loss, _ = solver.calculate_privacy_loss(schedule, attack_result)
        epsilon = max(privacy_loss) if privacy_loss else float("inf")
        twct = calculate_twct(schedule)
        return epsilon, twct

    def crossover(
        self, parent1: ParallelSchedule, parent2: ParallelSchedule
    ) -> ParallelSchedule:
        jobs_map = parent1.jobs
        mcount = self.params.machine_count
        all_jobs = list(jobs_map.values())
        allocation_tuples = [[] for _ in range(mcount)]

        for j in all_jobs:
            src_parent = parent1 if random.random() < 0.5 else parent2
            src_machine = next(
                (
                    mid
                    for mid, m in enumerate(src_parent.allocation)
                    if any(sj.id == j.id for sj in m)
                ),
                random.randrange(mcount),
            )
            allocation_tuples[src_machine].append((j.id, j.processing_time, j.weight))

        allocation_tuples = repair_and_fill(allocation_tuples, jobs_map)
        factory = ParallelScheduleFactory(self.params)
        return factory.from_job_tuple_list(allocation_tuples, schedule_type="wspt")

    def mutate(
        self, schedule: ParallelSchedule, mutation_rate: float = 0.25
    ) -> ParallelSchedule:
        factory = ParallelScheduleFactory(self.params)
        allocation_tuples = allocation_to_job_tuples(schedule.allocation, schedule.jobs)

        for _ in range(int(len(allocation_tuples) * 3)):
            if random.random() < mutation_rate:
                src = random.randrange(len(allocation_tuples))
                if allocation_tuples[src]:
                    job_idx = random.randrange(len(allocation_tuples[src]))
                    job = allocation_tuples[src].pop(job_idx)
                    dst = random.randrange(len(allocation_tuples))
                    insert_pos = random.randint(0, len(allocation_tuples[dst]))
                    allocation_tuples[dst].insert(insert_pos, job)

        for mid in range(len(allocation_tuples)):
            if random.random() < mutation_rate and len(allocation_tuples[mid]) > 2:
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
        population_size: int = 40,
        generations: int = 60,
        mutation_rate: float = 0.25,
    ) -> Tuple[float, float, ParallelSchedule]:
        population = [
            self.mutate(initial_schedule, mutation_rate=0.4)
            for _ in range(population_size)
        ]
        population[0] = initial_schedule

        best_schedule = initial_schedule
        best_eps, best_twct = self.calculate_metrics(best_schedule)
        logger.info(f"GA start: ε={best_eps:.4f}, TWCT={best_twct:.2f}")
        prev_best_twct = best_twct

        for g in range(generations):
            with ThreadPoolExecutor() as exe:
                metrics = list(exe.map(self.calculate_metrics, population))
            evaluated = list(zip(metrics, population))
            evaluated.sort(key=lambda x: (x[0][1], x[0][0]))

            best_eps, best_twct = evaluated[0][0]
            best_schedule = evaluated[0][1]
            logger.info(
                f"Gen {g + 1}/{generations} — ε={best_eps:.4f}, TWCT={best_twct:.2f}"
            )

            if abs(prev_best_twct - best_twct) < 1e-6:
                mutation_rate = min(0.6, mutation_rate * 1.4)
            else:
                mutation_rate = max(0.1, mutation_rate * 0.9)
            prev_best_twct = best_twct

            survivors = [s for (_, s) in evaluated[: max(2, population_size // 2)]]
            children = []
            while len(children) + len(survivors) < population_size:
                p1, p2 = random.sample(survivors, 2)
                child = self.crossover(p1, p2)
                child = self.mutate(child, mutation_rate)
                children.append(child)

            population = survivors + children

        logger.info(f"GA done: best TWCT={best_twct:.2f}, ε={best_eps:.4f}")
        return best_eps, best_twct, best_schedule


def run_genetic_for_schedule(args):
    schedule, params, idx = args
    optimizer = GeneticOptimizer(epsilon_threshold=0.7, params=params)
    logger.info(f"GA started for schedule {idx}")
    best_eps, best_twct, best_schedule = optimizer.genetic_algorithm(
        initial_schedule=schedule,
        population_size=40,
        generations=60,
        mutation_rate=0.25,
    )
    logger.info(f"GA finished for {idx}: ε={best_eps:.4f}, TWCT={best_twct:.2f}")

    res = {
        "index": idx,
        "best_epsilon": round(best_eps, 4),
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
    input_file = "new_balanced_schedules_30_3.json"
    schedules = load_schedules_from_json(input_file)
    tasks = [(s, p, i) for i, (s, p) in enumerate(schedules)]

    with ProcessPoolExecutor(max_workers=60) as exe:
        results = list(exe.map(run_genetic_for_schedule, tasks))

    out_file = "ga_30_3.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Ergebnis ist im {out_file}")


if __name__ == "__main__":
    main()
