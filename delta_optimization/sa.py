from dataclasses import dataclass
import logging
import random
import math
import json
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor

from privacyschedulingtools.total_weighted_completion_time.pup.solver.pup_parallel import (
    PupParallel,
)
from privacyschedulingtools.total_weighted_completion_time.entity.scheduling_parameters import (
    ParallelSchedulingParameters,
)
from privacyschedulingtools.total_weighted_completion_time.entity.parallel_schedule import (
    ParallelScheduleFactory,
    ParallelSchedule,
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
from privacyschedulingtools.total_weighted_completion_time.pup.util.transformations import (
    Transformation,
)

"""SA für opt von epsilon"""

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
    logger.addHandler(logging.FileHandler("sa_bad_epsilon_50_3.log"))


def load_schedules_from_json(
    file_path: str,
) -> List[Tuple[ParallelSchedule, ParallelSchedulingParameters]]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = [data]

    schedules = []
    for schedule_data in data:
        job_map = {}
        machines_data = schedule_data["machines"]
        machine_count = len(machines_data)

        for machine in machines_data:
            for job_info in machine["jobs"]:
                job_id = job_info["job_id"]
                if job_id not in job_map:
                    job_map[job_id] = Job(
                        job_id, job_info["processing_time"], job_info["weight"]
                    )

        jobs = list(job_map.values())
        processing_times = [j.processing_time for j in jobs]
        weights = [j.weight for j in jobs]

        params = ParallelSchedulingParameters(
            job_count=len(jobs),
            machine_count=machine_count,
            processing_time_domain=IntegerDomain(
                min(processing_times), max(processing_times)
            ),
            weight_domain=IntegerDomain(min(weights), max(weights)),
        )

        factory = ParallelScheduleFactory(params)
        schedule = factory.generate_schedule_from_jobs(
            jobs, machine_count, parameters=params, schedule_type="wspt"
        )
        schedules.append((schedule, params))

    return schedules


@dataclass
class Annealer:
    epsilon_threshold: float
    params: ParallelSchedulingParameters
    attack_repeats: int = 10

    def calculate_metrics(self, schedule: ParallelSchedule) -> Tuple[float, float]:
        adversary = ParallelLocalWSPTAdversary(self.params)
        epsilons = []
        for _ in range(self.attack_repeats):
            attack_result = adversary.execute_attack(schedule)
            solver = PupParallel(
                schedule,
                privacy_threshold=self.epsilon_threshold,
                utility_threshold=1.0,
                schedule_parameters=self.params,
            )
            privacy_loss, _ = solver.calculate_privacy_loss(schedule, attack_result)
            eps = max(privacy_loss) if privacy_loss else float("inf")
            epsilons.append(eps)
        avg_epsilon = sum(epsilons) / len(epsilons)
        twct = calculate_twct(schedule)
        return avg_epsilon, twct

    def get_neighbors(
        self,
        schedule: ParallelSchedule,
        transformations: List[Transformation],
        limit: int = 50,
    ) -> List[ParallelSchedule]:
        solver = PupParallel(
            schedule,
            privacy_threshold=self.epsilon_threshold,
            utility_threshold=1.0,
            schedule_parameters=self.params,
        )
        neighbors = []
        for t in random.sample(transformations, min(len(transformations), 3)):
            try:
                if t == Transformation.MOVE:
                    neighbors.extend(solver._add_move_neighbors(schedule))
                elif t == Transformation.SWAPPING_JOBS:
                    neighbors.extend(solver._add_swap_neighbors(schedule))
                elif t == Transformation.SWAP_PROC:
                    neighbors.extend(solver._add_swap_neighbors(schedule))
                    neighbors.extend(solver._add_processing_time_neighbors(schedule))
                elif t == Transformation.SWAP_ALL:
                    neighbors.extend(solver._add_swap_all(schedule))
                elif t == Transformation.MOVE_PROC:
                    neighbors.extend(solver._add_move_neighbors(schedule))
                    neighbors.extend(solver._add_processing_time_neighbors(schedule))
                elif t == Transformation.ALT_MOVE_PROC:
                    neighbors.extend(solver._add_move_neighbors_alt(schedule))
                    neighbors.extend(solver._add_processing_time_neighbors(schedule))
                elif t == Transformation.ALT_MOVE:
                    neighbors.extend(solver._add_move_neighbors_alt(schedule))
                elif t == Transformation.SWAP_ALL_PROC:
                    neighbors.extend(solver._add_swap_all(schedule))
                    neighbors.extend(solver._add_processing_time_neighbors(schedule))
            except Exception as e:
                logger.error(f"Neighbor error {t.name}: {str(e)}")

        if len(neighbors) > limit:
            neighbors = random.sample(neighbors, limit)
        return neighbors

    def simulated_annealing(
        self,
        initial_schedule: ParallelSchedule,
        transformations: List[Transformation],
        initial_temp: float = 100.0,
        cooling_rate: float = 0.95,
        min_temp: float = 0.1,
        max_iterations: int = 10,
    ) -> Tuple[float, float, ParallelSchedule]:
        current = initial_schedule
        best = current
        epsilon, twct = self.calculate_metrics(current)
        best_epsilon, best_twct = epsilon, twct

        logger.info(f"SA start: avg ε={epsilon:.4f}, TWCT={twct:.2f}")
        temperature = initial_temp

        for i in range(1, max_iterations + 1):
            neighbors = self.get_neighbors(current, transformations, limit=1)
            if not neighbors:
                temperature *= cooling_rate
                if temperature < min_temp:
                    break
                continue

            candidate = neighbors[0]
            cand_epsilon, cand_twct = self.calculate_metrics(candidate)

            if cand_twct > twct:
                continue

            delta_eps = cand_epsilon - epsilon
            accept = (cand_epsilon < epsilon) or random.random() < math.exp(
                -max(delta_eps, 0) / max(temperature, 1e-12)
            )

            if accept:
                current, epsilon, twct = candidate, cand_epsilon, cand_twct

            if cand_epsilon < best_epsilon:
                best, best_epsilon, best_twct = candidate, cand_epsilon, cand_twct
                logger.info(
                    f"[Iter {i}] NEW BEST: avg ε={best_epsilon:.4f}, TWCT={best_twct:.2f}"
                )

            temperature *= cooling_rate
            if temperature < min_temp:
                break

        return best_epsilon, best_twct, best


def run_sa_for_schedule(args):
    schedule, params, idx = args
    annealer = Annealer(epsilon_threshold=1.0, params=params)

    transformations = [t for t in Transformation]
    logger.info(f"=== SA gestartet für Schedule {idx} ===")
    best_epsilon, best_twct, best_schedule = annealer.simulated_annealing(
        initial_schedule=schedule, transformations=transformations, max_iterations=50
    )
    logger.info(
        f"=== SA fertig für Schedule {idx}: avg ε={best_epsilon:.4f}, TWCT={best_twct:.2f} ==="
    )

    result = {
        "index": idx,
        "avg_epsilon": round(best_epsilon, 4),
        "best_twct": round(best_twct, 2),
        "machines": [],
    }
    for machine_id, jobs in enumerate(best_schedule.allocation):
        machine_jobs = []
        for scheduled_job in jobs:
            job = best_schedule.jobs[scheduled_job.id]
            machine_jobs.append(
                {
                    "job_id": job.id,
                    "start_time": scheduled_job.start_time,
                    "processing_time": job.processing_time,
                    "weight": job.weight,
                }
            )
        result["machines"].append({"machine_id": machine_id, "jobs": machine_jobs})
    return result


def main():
    input_file = "bad_epsilon_50_3.json"
    output_file = "sa_bad_epsilon_50_3.json"

    schedules = load_schedules_from_json(input_file)
    tasks = [(s, p, i) for i, (s, p) in enumerate(schedules)]
    results = []

    with ProcessPoolExecutor(max_workers=60) as executor:
        for r in executor.map(run_sa_for_schedule, tasks):
            results.append(r)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Ergebnis ist in {output_file}")


if __name__ == "__main__":
    main()
