# MLOps Course

Learn how to combine machine learning with software engineering to develop, deploy and maintain production ML applications.

> MLOps concepts are interweaved and cannot be run in isolation, so be sure to complement the code in this repository with the detailed [MLOps lessons](https://madewithml.com/#mlops).

<div align="left">
    <a target="_blank" href="https://madewithml.com/"><img src="https://img.shields.io/badge/Subscribe-30K-brightgreen"></a>&nbsp;
    <a target="_blank" href="https://github.com/GokuMohandas/Made-With-ML"><img src="https://img.shields.io/github/stars/GokuMohandas/Made-With-ML.svg?style=social&label=Star"></a>&nbsp;
    <a target="_blank" href="https://www.linkedin.com/in/goku"><img src="https://img.shields.io/badge/style--5eba00.svg?label=LinkedIn&logo=linkedin&style=social"></a>&nbsp;
    <a target="_blank" href="https://twitter.com/GokuMohandas"><img src="https://img.shields.io/twitter/follow/GokuMohandas.svg?label=Follow&style=social"></a>
    <br>
</div>

- Lessons: https://madewithml.com/#mlops
- Code: [GokuMohandas/mlops-course](https://github.com/GokuMohandas/mlops-course)

<table>
	<tbody>
		<tr>
			<td><strong>🎨&nbsp; Design</strong></td>
			<td><strong>💻&nbsp; Developing</strong>&nbsp;</td>
			<td><strong>♻️&nbsp; Reproducibility</strong></td>
		</tr>
		<tr>
			<td><a href="https://madewithml.com/courses/mlops/design/">Product</a></td>
			<td><a href="https://madewithml.com/courses/mlops/packaging/">Packaging</a></td>
			<td><a href="https://madewithml.com/courses/mlops/git/">Git</a></td>
		</tr>
		<tr>
			<td><a href="https://madewithml.com/courses/mlops/design/#engineering">Engineering</a></td>
			<td><a href="https://madewithml.com/courses/mlops/organization/">Organization</a></td>
			<td><a href="https://madewithml.com/courses/mlops/pre-commit/">Pre-commit</a></td>
		</tr>
		<tr>
			<td><a href="https://madewithml.com/courses/mlops/design/#project-management">Project</a></td>
			<td><a href="https://madewithml.com/courses/mlops/logging/">Logging</a></td>
			<td><a href="https://madewithml.com/courses/mlops/versioning/">Versioning</a></td>
		</tr>
		<tr>
			<td><strong>🔢&nbsp; Data</strong></td>
			<td><a href="https://madewithml.com/courses/mlops/documentation/">Documentation</a></td>
			<td><a href="https://madewithml.com/courses/mlops/docker/">Docker</a></td>
		</tr>
		<tr style="height: 23.5px;">
			<td style="height: 23.5px;"><a href="https://madewithml.com/courses/mlops/exploratory-data-analysis/">Exploration</a></td>
			<td style="height: 23.5px;"><a href="https://madewithml.com/courses/mlops/styling/">Styling</a></td>
			<td style="height: 23.5px;"><strong>🚀&nbsp; Production</strong></td>
		</tr>
		<tr>
			<td><a href="https://madewithml.com/courses/mlops/labeling/">Labeling</a></td>
			<td><a href="https://madewithml.com/courses/mlops/makefile/">Makefile</a></td>
			<td><a href="https://madewithml.com/courses/mlops/dashboard/">Dashboard</a></td>
		</tr>
		<tr>
			<td><a href="https://madewithml.com/courses/mlops/preprocessing/">Preprocessing</a></td>
			<td><strong>📦&nbsp; Serving</strong></td>
			<td><a href="https://madewithml.com/courses/mlops/cicd/">CI/CD</a></td>
		</tr>
		<tr>
			<td><a href="https://madewithml.com/courses/mlops/splitting/">Splitting</a></td>
			<td><a href="https://madewithml.com/courses/mlops/cli/">Command-line</a></td>
			<td><a href="https://madewithml.com/courses/mlops/monitoring/">Monitoring</a></td>
		</tr>
		<tr>
			<td><a href="https://madewithml.com/courses/mlops/augmentation/">Augmentation</a></td>
			<td><a href="https://madewithml.com/courses/mlops/api/">RESTful API</a></td>
            <td><a href="https://madewithml.com/courses/mlops/systems-design/">Systems design</a></td>
		</tr>
		<tr>
			<td><strong>📈&nbsp; Modeling</strong></td>
			<td><strong>✅&nbsp; Testing</strong></td>
            <td><strong>⎈&nbsp; Data engineering</strong></td>
		</tr>
		<tr>
			<td>&nbsp;<a href="https://madewithml.com/courses/mlops/baselines/">Baselines</a></td>
			<td><a href="https://madewithml.com/courses/mlops/testing/">Code</a></td>
			<td><a href="https://madewithml.com/courses/mlops/data-stack/">Data stack</a></td>
		</tr>
		<tr>
			<td><a href="https://madewithml.com/courses/mlops/evaluation/">Evaluation</a></td>
			<td><a href="https://madewithml.com/courses/mlops/testing/#data">Data</a></td>
            <td><a href="https://madewithml.com/courses/mlops/orchestration/">Orchestration</a></td>
		</tr>
		<tr>
			<td><a href="https://madewithml.com/courses/mlops/experiment-tracking/">Experiment tracking</a></td>
			<td><a href="https://madewithml.com/courses/mlops/testing/#models">Models</a></td>
			<td><a href="https://madewithml.com/courses/mlops/feature-store/">Feature store</a></td>
		</tr>
		<tr>
			<td><a href="https://madewithml.com/courses/mlops/optimization/">Optimization</a></td>
			<td>&nbsp;</td>
			<td>&nbsp;</td>
		</tr>
	</tbody>
</table>

### Virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -e ".[dev]"
pre-commit install
pre-commit autoupdate
```

> If the commands above do not work, please refer to the [packaging](https://madewithml.com/courses/mlops/packaging/) lesson. We highly recommend using [Python version](https://madewithml.com/courses/mlops/packaging/#python) `3.9.1`.

### Directory
```bash
tagifai/
├── data.py       - data processing components
├── evaluate.py   - evaluation components
├── main.py       - training/optimization operations
├── predict.py    - inference components
├── train.py      - training components
└── utils.py      - supplementary utilities
```

### Workflow
```bash
python tagifai/main.py elt-data
python tagifai/main.py optimize --args-fp="config/args.json" --study-name="optimization" --num-trials=10
python tagifai/main.py train-model --args-fp="config/args.json" --experiment-name="baselines" --run-name="sgd"
python tagifai/main.py predict-tag --text="Transfer learning with transformers for text classification."
```

### API
```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload --reload-dir tagifai --reload-dir app  # dev
gunicorn -c app/gunicorn.py -k uvicorn.workers.UvicornWorker app.api:app  # prod
```

<hr>
<!-- Citation -->
To cite this content, please use:

```bibtex
@misc{madewithml,
    author       = {Goku Mohandas},
    title        = {MLOps Course - Made With ML},
    howpublished = {\url{https://madewithml.com/}},
    year         = {2022}
}
```
