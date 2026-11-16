# Ongoing work
- see /home/ub/code/det3d/docs/integration/ i am integrating nndetection code into det3d, and nnunet into fran.
RULE OF THUMB: reuse existing well supported frameworks over my own snippets.
Keep adaptability to custom my own pipelines and also adaptability to flexible hyperparameter search orchestrated by agents. For example, if nndetection is given total sole control over deciding depth etc., hyperparameter search algos wont do much. However, it can be a good baseline to improve upon. So while writing code, if it is OBVIOUS and convenient to do so, hooks should be kept to allow extensibility to nas and hyperparameter search.
Once integration is cimpleted, this section should be deleted as well as integration/ folder.
integration work that invvoles both nnunet and nndetection: fran resues some nnunet concepts - it might be better to kill 2 birds 1 stone in delivering this integration and the way to do that is to work primrariy on detection and signposting how fran might benefit from changes too, and whether its better to do a single change upstream that works across libraries



# Repo-specific instructions
- Read repo rules from ~/code/fran/AGENTS.md
- conda env: `dl`
- CLI toolbox: [FUNCTIONS.md](FUNCTIONS.md); global index [~/code/agent/CATALOG.md](../agent/CATALOG.md)
- **Inference alignment debug:** see [det3d/extra/infer_debug/README.md](det3d/extra/infer_debug/README.md); rule `.cursor/rules/infer-debug-suite.mdc`
