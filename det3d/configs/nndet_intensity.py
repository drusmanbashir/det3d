"""Plan CT intensity normalization from the detection foreground."""

from collections import defaultdict

from fastcore.basics import listify
from fran.configs.remapping import parse_excel_datasources, parse_nested_remapping
from fran.data.dataregistry import DS
from fran.preprocessing.helpers import import_h5py
from fran.preprocessing.hu_histogram import HUHistogram


def selected_detection_cases(project, plan):  #AI
    """Return the non-test cases in the configured detection datasources."""
    names = parse_excel_datasources(plan["datasources"])
    specs = [DS[name] for name in names]
    datasets = {spec.ds for spec in specs}
    if len(datasets) != len(names) or len({spec.name for spec in specs}) != len(names):
        raise ValueError(f"Plan {plan['plan_name']} has ambiguous detection datasources")
    rows = project.sql_query("SELECT ds, case_id FROM datasources WHERE test=0 ORDER BY ds, case_id")
    return [(ds, str(case_id)) for ds, case_id in rows if ds in datasets]


def _case_foreground(labels, source, bounded, ignored):  #AI
    """Map raw histogram rows to labels retained after both preprocessing stages."""
    drop = []
    foreground = set()
    for raw in labels:
        intermediate = source.get(int(raw), int(raw))
        final = bounded.get(intermediate, intermediate)
        if final <= 0 or final in ignored:
            drop.append(raw)
        else:
            foreground.add(int(final))
    return drop, foreground


def plan_intensity_properties(project, plan):  #AI
    """Pool native rounded-HU bins for detection labels and return stats and class IDs."""
    if plan.get("imported_folder"):
        raise ValueError(f"Plan {plan['plan_name']} has ambiguous detection foreground from imported labels")
    names = parse_excel_datasources(plan["datasources"])
    source = parse_nested_remapping(plan, "remapping_source", as_dict=True) or [None] * len(names)
    bounded = parse_nested_remapping(plan, "remapping_lbd_rbd", as_dict=True) or [None] * len(names)
    remaps = {DS[name].ds: (first or {}, second or {})
              for name, first, second in zip(names, source, bounded)}
    ignored = plan.get("ignore_labels_cc")
    if ignored is None:
        ignored = plan.get("ignore_labels")
    ignored = set(listify(ignored))
    by_dataset = defaultdict(list)
    for dataset, case_id in selected_detection_cases(project, plan):
        by_dataset[dataset].append(case_id)
    h5py = import_h5py()
    pooled = None
    foreground = set()
    for dataset, case_ids in by_dataset.items():
        paths = [row["h5_fname"] for row in project.global_properties["datasources"] if row["ds"] == dataset]
        if len(paths) != 1 or len(case_ids) != len(set(case_ids)):
            raise ValueError(f"Plan {plan['plan_name']} has ambiguous histogram inputs for {dataset}")
        first, second = remaps[dataset]
        with h5py.File(paths[0], "r") as archive:
            for case_id in case_ids:
                case = archive[case_id]
                if "hist_labels" not in case.attrs:
                    raise ValueError(f"{paths[0]} has no per-label histograms; rerun Datasource.process")
                labels = case.attrs["hist_labels"]
                drop, kept = _case_foreground(labels, first, second, ignored)
                if kept:
                    histogram = HUHistogram.from_label_rows(case[:], labels, case.attrs["hu_min"], drop)
                    pooled = histogram if pooled is None else pooled + histogram
                    foreground.update(kept)
    if pooled is None or not pooled.counts.any():
        raise ValueError(f"Plan {plan['plan_name']} has empty detection foreground")
    stats = pooled.stats([0.5, 99.5])
    lo, hi = stats["percentiles"]
    if stats["std"] <= 0 or lo >= hi:
        raise ValueError(f"Plan {plan['plan_name']} has degenerate foreground intensity statistics")
    properties = {"mean": stats["mean"], "sd": stats["std"], "percentile_00_5": lo, "percentile_99_5": hi}
    return properties, sorted(foreground)
