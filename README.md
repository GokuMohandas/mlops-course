<div align="center">
<h1><img width="30" src="https://madewithml.com/static/images/rounded_logo.png">&nbsp;<a href="https://madewithml.com/">Made With ML</a></h1>
Applied ML · MLOps · Production
<br>
Join 20K+ developers in learning how to responsibly <a href="https://madewithml.com/about/">deliver value</a> with applied ML.
</div>

<br>

<div align="center">
    <a target="_blank" href="https://madewithml.com/subscribe/"><img src="https://img.shields.io/badge/Subscribe-20K-brightgreen"></a>&nbsp;
    <a target="_blank" href="https://github.com/GokuMohandas/madewithml"><img src="https://img.shields.io/github/stars/GokuMohandas/madewithml.svg?style=social&label=Star"></a>&nbsp;
    <a target="_blank" href="https://www.linkedin.com/in/goku"><img src="https://img.shields.io/badge/style--5eba00.svg?label=LinkedIn&logo=linkedin&style=social"></a>&nbsp;
    <a target="_blank" href="https://twitter.com/GokuMohandas"><img src="https://img.shields.io/twitter/follow/GokuMohandas.svg?label=Follow&style=social"></a>
</div>

<br>

> If you need refresh yourself on ML algorithms, check our out [ML Foundations](https://github.com/GokuMohandas/madewithml) repository (🔥&nbsp; Among the <a href="https://github.com/topics/deep-learning" target="_blank">top ML</a> repositories on GitHub)

<br>

<table>
    <tr>
        <td align="center"><b>📦&nbsp; Product</b></td>
        <td align="center"><b>🔢&nbsp; Data</b></td>
        <td align="center"><b>📈&nbsp; Modeling</b></td>
    </tr>
    <tr>
        <td><a href="https://madewithml.com/courses/applied-ml/objective/">Objective</a></td>
        <td><a href="https://madewithml.com/courses/applied-ml/annotation/">Annotation</a></td>
        <td><a href="https://madewithml.com/courses/applied-ml/baselines/">Baselines</a></td>
    </tr>
    <tr>
        <td><a href="https://madewithml.com/courses/applied-ml/solution/">Solution</a></td>
        <td><a href="https://madewithml.com/courses/applied-ml/exploratory-data-analysis/">Exploratory data analysis</a></td>
        <td><a href="https://madewithml.com/courses/applied-ml/experiment-tracking/">Experiment tracking</a></td>
    </tr>
    <tr>
        <td><a href="https://madewithml.com/courses/applied-ml/evaluation/">Evaluation</a></td>
        <td><a href="https://madewithml.com/courses/applied-ml/splitting/">Splitting</a></td>
        <td><a href="https://madewithml.com/courses/applied-ml/optimization/">Optimization</a></td>
    </tr>
    <tr>
        <td><a href="https://madewithml.com/courses/applied-ml/iteration/">Iteration</a></td>
        <td><a href="https://madewithml.com/courses/applied-ml/preprocessing/">Preprocessing</a></td>
        <td></td>
    </tr>
</table>

<table>
    <tr>
        <td align="center"><b>📝&nbsp; Scripting</b></td>
        <td align="center"><b>✅&nbsp; Testing</b></td>
        <td align="center"><b>📦&nbsp; Application</b></td>
    </tr>
    <tr>
        <td><a href="https://madewithml.com/courses/applied-ml/organization/">Organization</a></td>
        <td>Testing <small>(code)</small></td>
        <td>RESTful API</td>
    </tr>
    <tr>
        <td>Documentation</td>
        <td>Testing <small>(data)</small></td>
        <td>Databases</td>
    </tr>
    <tr>
        <td>Logging</td>
        <td>Testing <small>(model)</small></td>
        <td>Authentication</td>
    </tr>
    <tr>
        <td>Styling</td>
        <td></td>
        <td></td>
    </tr>
</table>

<table>
    <tr>
        <td align="center"><b>⏰&nbsp; Version control</b></td>
        <td align="center"><b>🚀&nbsp; Production</b></td>
        <td align="center"><b>(cont.)</b></td>
    </tr>
    <tr>
        <td>Git</td>
        <td>Dashboard</td>
        <td>Serving</td>
    </tr>
    <tr>
        <td>Precommit</td>
        <td>Docker</td>
        <td>Feature stores</td>
    </tr>
    <tr>
        <td>Versioning</td>
        <td>CI/CD</td>
        <td>Workflow management</td>
    </tr>
    <tr>
        <td></td>
        <td>Monitoring</td>
        <td>Active learning</td>
    </tr>
</table>

📆&nbsp; new lesson every week!<br>
<a href="https://madewithml.com/subscribe/" target="_blank">Subscribe</a> for our monthly updates on new content.

<br>

## Set up
```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
make install-dev
```

## Start Jupyterlab
```bash
python -m ipykernel install --user --name=tagifai
jupyter labextension install @jupyter-widgets/jupyterlab-manager
jupyter labextension install @jupyterlab/toc
jupyter lab
```
> You can also run all notebooks on [Google Colab](https://colab.research.google.com/github/GokuMohandas/applied-ml/blob/main/notebooks/tagifai.ipynb).

## Actions
```
tagifai --help
```

## Directory structure
```
tagifai/
├── config.py     - configuration setup
├── data.py       - data processing utilities
├── main.py       - main operations (CLI)
├── models.py     - model architectures
├── predict.py    - inference utilities
├── train.py      - training utilities
└── utils.py      - supplementary utilities
```

## MLFlow
```
mlflow server -h 0.0.0.0 -p 5000 --backend-store-uri assets/experiments/
```

## Mkdocs
```
python -m mkdocs serve
```

## FAQ

### Why is this free?
While this content is for everyone, it's especially targeted towards people who don't have as much opportunity to learn. I firmly believe that creativity and intelligence are randomly distributed but opportunity is siloed. I want to enable more people to create and contribute to innovation.

### Who is the author?
- I've deployed large scale ML systems at Apple as well as smaller systems with constraints at startups and want to share the common principles I've learned along the way.
- I created [Made With ML](https://madewithml.com/) so that the community can explore, learn and build ML and I learned how to build it into an end-to-end product that's currently used by over 20K monthly active users.
- Connect with me on <a href="https://twitter.com/GokuMohandas" target="_blank"><i class="fab fa-twitter ai-color-info mr-1"></i>Twitter</a> and <a href="https://www.linkedin.com/in/goku" target="_blank"><i class="fab fa-linkedin ai-color-primary mr-1"></i>LinkedIn</a>
