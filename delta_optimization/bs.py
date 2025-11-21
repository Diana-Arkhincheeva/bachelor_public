from dataclasses import dataclass
import logging
import json
import random
from typing import List, Tuple, Optional
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


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
    logger.addHandler(logging.FileHandler("bs_delta_50_3.log"))


def load_schedules_from_json(
    file_path: str,
) -> List[Tuple[ParallelSchedule, ParallelSchedulingParameters]]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = [data]

    schedules: List[Tuple[ParallelSchedule, ParallelSchedulingParameters]] = []
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
        processing_times = [job.processing_time for job in jobs]
        weights = [job.weight for job in jobs]

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
class BeamOptimizer:
    epsilon_threshold: float
    params: ParallelSchedulingParameters
    attack_repeats: int = 10

    def calculate_metrics(self, schedule: ParallelSchedule) -> Tuple[float, float]:
        adversary = ParallelLocalWSPTAdversary(self.params)
        success_count = 0

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

            if eps >= 1.0:
                success_count += 1

        avg_epsilon = success_count / float(self.attack_repeats)
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

        neighbors: List[ParallelSchedule] = []
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

        neighbors = [n for n in neighbors if n is not schedule]
        if len(neighbors) > limit:
            neighbors = random.sample(neighbors, limit)
        return neighbors

    def beam_search(
        self,
        initial_schedule: ParallelSchedule,
        transformations: List[Transformation],
        beam_width: int = 5,
        max_iterations: int = 50,
        neighbor_limit: int = 50,
        output_file: Optional[str] = None,
    ) -> Tuple[float, float, ParallelSchedule]:
        epsilon, twct = self.calculate_metrics(initial_schedule)
        beam = [(epsilon, twct, initial_schedule)]
        best_epsilon, best_twct, best_schedule = epsilon, twct, initial_schedule
        valid_schedules = []

        logger.info(f"Beam Search gestartet: start ε={epsilon:.4f}, TWCT={twct:.2f}")

        for it in range(max_iterations):
            logger.info(f"Iteration {it + 1}, Beam size={len(beam)}")
            all_candidates = []
            for _, _, schedule in beam:
                neighbors = self.get_neighbors(
                    schedule, transformations, limit=neighbor_limit
                )
                for n in neighbors:
                    eps, tw = self.calculate_metrics(n)
                    all_candidates.append((n, (eps, tw)))

            if not all_candidates:
                logger.info("Keine Kandidaten gefunden")
                break

            for schedule, (eps, tw) in all_candidates:
                if eps <= self.epsilon_threshold:
                    valid_schedules.append(
                        {"epsilon": round(eps, 4), "twct": round(tw, 2)}
                    )
            all_candidates.sort(key=lambda x: (x[1][0], x[1][1]))
            top_k = all_candidates[:beam_width]
            beam = [(eps, tw, sch) for sch, (eps, tw) in top_k]

            eps0, tw0, sch0 = beam[0]
            if (eps0 < best_epsilon) or (eps0 == best_epsilon and tw0 < best_twct):
                best_epsilon, best_twct, best_schedule = eps0, tw0, sch0
                logger.info(
                    f"[Iter {it + 1}] UPDATE best: ε={best_epsilon:.4f}, TWCT={best_twct:.2f}"
                )

            if best_epsilon == 0.0:
                logger.info("Bestes mögliches ε=0 erreicht.")
                break

        if output_file and valid_schedules:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(valid_schedules, f, indent=2, ensure_ascii=False)
            logger.info(
                f"Gespeichert {len(valid_schedules)} gültige Schedules in {output_file}"
            )

        return best_epsilon, best_twct, best_schedule


def run_beam_for_schedule(args):
    schedule, params, idx = args
    optimizer = BeamOptimizer(epsilon_threshold=0.7, params=params)

    transformations = [t for t in Transformation]

    logger.info(f"Beam Search gestartet für Kalender {idx}")
    best_epsilon, best_twct, best_schedule = optimizer.beam_search(
        initial_schedule=schedule,
        transformations=transformations,
        beam_width=5,
        max_iterations=10,
        neighbor_limit=30,
        output_file=None,
    )
    logger.info(
        f"Beam Search fertig für Kalender {idx}, ε={best_epsilon:.4f}, TWCT={best_twct:.2f}"
    )

    result_dict = {
        "index": idx,
        "best_epsilon": round(best_epsilon, 4),
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
        result_dict["machines"].append({"machine_id": machine_id, "jobs": machine_jobs})

    return result_dict


def main():
    input_file = "bad_epsilon_50_3.json"
    output_file = "bs_bad_epsilon_50_3.json"

    schedules = load_schedules_from_json(input_file)
    tasks = [(s, p, i) for i, (s, p) in enumerate(schedules)]

    with ProcessPoolExecutor(max_workers=60) as executor:
        results = list(executor.map(run_beam_for_schedule, tasks))

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Ergebnis ist in {output_file}")


if __name__ == "__main__":
    main()
