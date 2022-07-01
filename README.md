# MLOps Course

Learn how to apply ML to build a production grade product to deliver value.

- Lessons: https://madewithml.com/#mlops
- Code: [GokuMohandas/mlops-course](https://github.com/GokuMohandas/mlops-course)

<table>
	<tbody>
		<tr>
			<td><strong>📦&nbsp; Purpose</strong></td>
			<td><strong>💻&nbsp; Developing</strong>&nbsp;</td>
			<td><strong>♻️&nbsp; Reproducibility</strong></td>
		</tr>
		<tr>
			<td><a href="https://madewithml.com/courses/mlops/purpose/">Product</a></td>
			<td><a href="https://madewithml.com/courses/mlops/packaging/">Packaging</a></td>
			<td><a href="https://madewithml.com/courses/mlops/git/">Git</a></td>
		</tr>
		<tr>
			<td><a href="https://madewithml.com/courses/mlops/product/#system-design">System design</a></td>
			<td><a href="https://madewithml.com/courses/mlops/organization/">Organization</a></td>
			<td><a href="https://madewithml.com/courses/mlops/pre-commit/">Pre-commit</a></td>
		</tr>
		<tr>
			<td><a href="https://madewithml.com/courses/mlops/purpose#project-management">Project</a></td>
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
			<td><a href="https://madewithml.com/courses/mlops/cicd/">CI/CD workflows</a></td>
		</tr>
		<tr>
			<td><a href="https://madewithml.com/courses/mlops/splitting/">Splitting</a></td>
			<td><a href="https://madewithml.com/courses/mlops/cli/">Command-line</a></td>
			<td><a href="https://madewithml.com/courses/mlops/infrastructure/">Infrastructure</a></td>
		</tr>
		<tr>
			<td><a href="https://madewithml.com/courses/mlops/augmentation/">Augmentation</a></td>
			<td><a href="https://madewithml.com/courses/mlops/api/">RESTful API</a></td>
			<td><a href="https://madewithml.com/courses/mlops/monitoring/">Monitoring</a></td>
		</tr>
		<tr>
			<td><strong>📈&nbsp; Modeling</strong></td>
			<td><strong>✅&nbsp; Testing</strong></td>
			<td><a href="https://madewithml.com/courses/mlops/feature-store/">Feature store</a></td>
		</tr>
		<tr>
			<td>&nbsp;<a href="https://madewithml.com/courses/mlops/baselines/">Baselines</a></td>
			<td><a href="https://madewithml.com/courses/mlops/testing/">Code</a></td>
			<td><a>Data stack</a>&nbsp;<small>(Aug 2022)</small></td>
		</tr>
		<tr>
			<td><a href="https://madewithml.com/courses/mlops/evaluation/">Evaluation</a></td>
			<td><a href="https://madewithml.com/courses/mlops/testing/#data">Data</a></td>
			<td><a>Orchestration</a>&nbsp;<small>(Aug 2022)</small></td>
		</tr>
		<tr>
			<td><a href="https://madewithml.com/courses/mlops/experiment-tracking/">Experiment tracking</a></td>
			<td><a href="https://madewithml.com/courses/mlops/testing/#models">Models</a></td>
			<td>&nbsp;</td>
		</tr>
		<tr>
			<td><a href="https://madewithml.com/courses/mlops/optimization/">Optimization</a></td>
			<td>&nbsp;</td>
			<td>&nbsp;</td>
		</tr>
	</tbody>
</table>

📆&nbsp; More content coming soon!<br>
<a href="https://newsletter.madewithml.com" target="_blank">Subscribe</a> for our monthly updates on new content.

### Instructions

We highly recommend going through the [lessons](https://madewithml.com/#mlops) one at a time and building the code base as we progress. For every concept, we focus on the fundamentals and then dive into the code, at which point we can refer to this repository as a guide.

### Virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -e ".[dev]"
pre-commit install
pre-commit autoupdate
```

> If the commands above do not work, please refer to the [packaging](https://madewithml.com/courses/mlops/packaging/) lesson. We highly recommend using [Python version](https://madewithml.com/courses/mlops/packaging/#python) `3.7.13`.

### Directory
```bash
tagifai/
├── data.py       - data processing utilities
├── evaluate.py   - evaluation components
├── main.py       - training/optimization operations
├── predict.py    - inference utilities
├── train.py      - training utilities
└── utils.py      - supplementary utilities
```

### Workflow
```bash
python tagifai/main.py load-data
python tagifai/main.py label-data --args-fp="config/args.json"
python tagifai/main.py optimize --args-fp="config/args.json" --study-name="optimization" --num-trials=10
python tagifai/main.py train-model --args-fp="config/args.json" --experiment-name="baselines" --run-name="sgd"
python tagifai/main.py predict-tag --text="Transfer learning with transformers for text classification."
```

### API
```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload --reload-dir tagifai --reload-dir app  # dev
gunicorn -c app/gunicorn.py -k uvicorn.workers.UvicornWorker app.api:app  # prod
```
