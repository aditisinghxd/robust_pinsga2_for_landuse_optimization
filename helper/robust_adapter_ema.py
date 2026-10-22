
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
import numpy as np

from ema_workbench import (Model, RealParameter, ScalarOutcome, Constant,
                               Policy, Scenario)


# Type aliases
Candidate = Union[Sequence[float], np.ndarray]
Metrics = Dict[str, Any]
Mapper = Callable[[Candidate], Mapping[str, float]]  # candidate -> lever dict


@dataclass
class RobustnessConfig:
    n_scenarios: int = 250
    use_multiprocessing: bool = True
    seed: Optional[int] = None


class EmaRobustnessAdapter:
    """
    Evaluate robustness of a *candidate* by running your model across many scenarios via EMA Workbench.
    You provide:
      - model_function: callable(c=..., z=..., fet=..., **levers) -> dict of outcomes
      - candidate_to_policy: mapper turning candidate vectors into lever dictionaries
    We return a dict of robustness metrics.
    """

    def __init__(
        self,
        model_function: Callable[..., Mapping[str, float]],
        candidate_to_policy: Mapper,
        uncertainties: Optional[List[RealParameter]] = None,
        outcomes: Optional[List[ScalarOutcome]] = None,
        constants: Optional[List[Constant]] = None,
        config: Optional[RobustnessConfig] = None,
    ):
        '''if not EMA_AVAILABLE:
            raise ImportError(
                "ema_workbench is required for EmaRobustnessAdapter. "
                "Install with: pip install ema_workbench"
            )'''

        self.model = Model("RobustModel", function=model_function)
        self.model.uncertainties = uncertainties or [
            RealParameter("c", 1.0, 10.0),
            RealParameter("z", 0.1, 0.4),
            RealParameter("fet", 0.5, 2.0),
        ]
        self.model.outcomes = outcomes or [
            ScalarOutcome("S"),
            ScalarOutcome("Yield"), 
        ]
        self.model.constants = constants or []

        self.candidate_to_policy: Mapper = candidate_to_policy
        self.cfg = config or RobustnessConfig()

        # Seed numpy for scenario reproducibility (EMA will draw from np.random)
        if self.cfg.seed is not None:
            np.random.seed(self.cfg.seed)

    def _compute_metrics(self, out: Dict[str, np.ndarray]) -> Metrics:
        # Outcomes are returned as arrays over scenarios
        S = np.asarray(out.get("S", []), dtype=float)
        Y = np.asarray(out.get("Yield", []), dtype=float)

        metrics: Metrics = {}

        # simple regret vs best scenario-wise (maximize S and Yield)
        if S.size:
            best_S = np.max(S)
            regret_S = best_S - S

            metrics["regret_S_max"] = float(np.max(regret_S))
            metrics["regret_S_p90"] = float(np.percentile(regret_S, 90))
            metrics["S_median"] = float(np.percentile(S, 50))  # or np.median(S)
            metrics["S_p90"] = float(np.percentile(S, 90))                 
        else:
            metrics["regret_S_max"] = float("nan")
            metrics["regret_S_p90"] = float("nan")
            metrics["S_median"] = float("nan")
            metrics["S_p90"] = float("nan")


        if Y.size:
            best_Y = np.max(Y)
            regret_Y = best_Y - Y

            metrics["regret_Yield_max"] = float(np.max(regret_Y))
            metrics["regret_Yield_p90"] = float(np.percentile(regret_Y, 90))
            metrics["Yield_median"] = float(np.percentile(Y, 50))  # or np.median(Y)
            metrics["Yield_p90"] = float(np.percentile(Y, 90))                 
        else:
            metrics["regret_Yield_max"] = float("nan")
            metrics["regret_Yield_p90"] = float("nan")
            metrics["Yield_median"] = float("nan")
            metrics["Yield_p90"] = float("nan")



        return metrics

    def __call__(self, candidate: Candidate, policy_name: str = "cand") -> Metrics:
        """Evaluate robustness metrics for a single candidate."""
        lever_values = dict(self.candidate_to_policy(candidate))
        
        from ema_workbench import CategoricalParameter
        import numpy as np
        lu = np.asarray(lever_values["lu_map"])
        lu_dummy = lu.copy()
        self.model.levers = [CategoricalParameter("lu_map", [lu, lu_dummy])]

        policy = Policy(policy_name, lu_map=lu)
        
        from ema_workbench.em_framework.evaluators import SequentialEvaluator

        Evaluator = SequentialEvaluator

        # Evaluate a single policy across many scenarios
        if self.cfg.use_multiprocessing:
            try:
                from ema_workbench.em_framework.evaluators import MultiprocessingEvaluator
                Evaluator = MultiprocessingEvaluator
            except ImportError:
                # Your EMA version doesn't have it -> fall back to sequential
                Evaluator = SequentialEvaluator
        
        with Evaluator(self.model) as ev:
            experiments, outcomes = ev.perform_experiments(
                scenarios=self.cfg.n_scenarios, policies=[policy]
            )
        #print(outcomes)
        metrics = self._compute_metrics(outcomes)

        # EMA returns outcomes as {name: np.ndarray shape (n_exp,)} for single policy
        return metrics, experiments, outcomes
        #return metrics

    # Batch API
    def map(self, candidates: Iterable[Candidate]) -> List[Metrics]:
        return [self(c) for c in candidates]
