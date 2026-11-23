"""Real spawn-process workers with closed JSON IPC and deterministic ownership."""

from __future__ import annotations

import multiprocessing as mp
import os
from contextlib import suppress
from multiprocessing.connection import Connection, wait
from multiprocessing.process import BaseProcess
from time import perf_counter
from typing import Any

from policytwin.coordination.plan import (
    CoordinationExecutionPlan,
    RunSpec,
    compile_contract,
)
from policytwin.coordination.schema import (
    CoordinationError,
    TaskRequest,
    TaskResult,
    integer,
)
from policytwin.coordination.workload import compute_task
from policytwin.provenance import canonical_json_bytes, read_json_bytes_object


def _worker(
    spec_data: dict[str, Any],
    worker_id: int,
    incoming: Connection,
    outgoing: Connection,
) -> None:
    spec = RunSpec.from_mapping(spec_data)
    contract = compile_contract(spec)
    outgoing.send_bytes(
        canonical_json_bytes({"worker_id": worker_id, "pid": os.getpid()})
    )
    try:
        while True:
            data = read_json_bytes_object(incoming.recv_bytes())
            if data == {"operation": "stop"}:
                return
            started = perf_counter()
            try:
                request = TaskRequest.from_mapping(data["request"])
                collect_draws = data["collect_draws"]
                mutate = data["mutate_quantity"]
                if type(collect_draws) is not bool or type(mutate) is not bool:
                    raise CoordinationError("worker switches must be Boolean")
                result = compute_task(
                    spec,
                    contract,
                    request,
                    worker_id,
                    os.getpid(),
                    collect_draws=collect_draws,
                )
                if mutate:
                    from policytwin.coordination.faults import mutate_quantity

                    result = mutate_quantity(result)
                response = {
                    "result": result.to_mapping(),
                    "compute_seconds": perf_counter() - started,
                    "error": None,
                }
            except Exception as error:
                response = {
                    "result": None,
                    "compute_seconds": perf_counter() - started,
                    "error": f"{type(error).__name__}: {error}",
                }
            outgoing.send_bytes(canonical_json_bytes(response))
    except EOFError:
        return
    finally:
        incoming.close()
        outgoing.close()


class WorkerPool:
    """Own and clean up the declared number of separate agent-compute processes."""

    def __init__(self, spec: RunSpec, plan: CoordinationExecutionPlan) -> None:
        self.spec = spec
        self.plan = plan
        self.pipes: list[Connection] = []
        self.outgoing: list[Connection] = []
        self.processes: list[BaseProcess] = []
        self.pids: dict[int, int] = {}
        self.compute_seconds = 0.0
        started = perf_counter()
        context = mp.get_context("spawn")
        try:
            for worker_id in range(plan.worker_count):
                # Two anonymous OS pipes avoid socket-based duplex IPC entirely.
                child_in, parent_out = context.Pipe(duplex=False)
                parent_in, child_out = context.Pipe(duplex=False)
                process = context.Process(
                    target=_worker,
                    args=(spec.to_mapping(), worker_id, child_in, child_out),
                    daemon=False,
                )
                process.start()
                child_in.close()
                child_out.close()
                self.pipes.append(parent_in)
                self.outgoing.append(parent_out)
                self.processes.append(process)
            for worker_id, pipe in enumerate(self.pipes):
                if not pipe.poll(30):
                    raise CoordinationError("worker initialization watchdog expired")
                hello = read_json_bytes_object(pipe.recv_bytes())
                if hello["worker_id"] != worker_id:
                    raise CoordinationError("worker handshake identity mismatch")
                self.pids[worker_id] = integer(hello["pid"], "worker pid", 1)
        except BaseException:
            self.close()
            raise
        self.startup_seconds = perf_counter() - started

    def execute(
        self,
        jobs: list[tuple[int, TaskRequest]],
        *,
        collect_draws: bool = True,
        quantity_mutation_tasks: frozenset[str] = frozenset(),
    ) -> list[TaskResult]:
        """Dispatch at most one pending request per worker; retain arrival order."""
        queues: dict[int, list[TaskRequest]] = {
            i: [] for i in range(self.plan.worker_count)
        }
        for owner, request in jobs:
            queues[owner].append(request)
        pending: dict[Connection, int] = {}

        def send_next(owner: int) -> None:
            if queues[owner]:
                request = queues[owner].pop(0)
                pipe = self.pipes[owner]
                self.outgoing[owner].send_bytes(
                    canonical_json_bytes(
                        {
                            "request": request.to_mapping(),
                            "collect_draws": collect_draws,
                            "mutate_quantity": request.task_id
                            in quantity_mutation_tasks,
                        }
                    )
                )
                pending[pipe] = owner

        for owner in queues:
            send_next(owner)
        results: list[TaskResult] = []
        while pending:
            ready = wait(list(pending), timeout=30)
            if not ready:
                raise CoordinationError("worker response watchdog expired")
            for pipe in ready:
                if not isinstance(pipe, Connection):
                    raise CoordinationError("unexpected IPC wait handle")
                owner = pending.pop(pipe)
                response = read_json_bytes_object(pipe.recv_bytes())
                if response["error"] is not None:
                    raise CoordinationError(str(response["error"]))
                result = TaskResult.from_mapping(response["result"])
                if result.worker_id != owner or result.worker_pid != self.pids[owner]:
                    raise CoordinationError("IPC endpoint/result identity mismatch")
                results.append(result)
                duration = response["compute_seconds"]
                if (
                    not isinstance(duration, float | int)
                    or isinstance(duration, bool)
                    or duration < 0
                ):
                    raise CoordinationError("invalid worker compute duration")
                self.compute_seconds += duration
                send_next(owner)
        return results

    def close(self) -> None:
        """Gracefully stop workers, then terminate only this pool's stuck children."""
        for pipe in self.outgoing:
            with suppress(OSError, BrokenPipeError, EOFError):
                pipe.send_bytes(canonical_json_bytes({"operation": "stop"}))
        for process in self.processes:
            process.join(timeout=2)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
        for pipe in (*self.pipes, *self.outgoing):
            pipe.close()

    def __enter__(self) -> WorkerPool:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
