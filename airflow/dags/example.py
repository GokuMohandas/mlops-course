from airflow.decorators import dag, task
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago

# Default DAG args
default_args = {
    "owner": "airflow",
}


# Define DAG
@dag(
    dag_id="example1",
    description="Example DAG with task decorators",
    default_args=default_args,
    schedule_interval=None,
    start_date=days_ago(2),
    tags=["example"],
)
def example_1():
    @task
    def task_1():
        return 1

    @task
    def task_2(x):
        return x + 1

    x = task_1()
    y = task_2(x=x)  # NOQA: F841 (assigned by unused)


def _task_1(ti):
    x = 2
    ti.xcom_push(key="x", value=x)


def _task_2(ti):
    x = ti.xcom_pull(key="x", task_ids=["task_1"])[0]
    y = x + 3
    ti.xcom_push(key="y", value=y)


@dag(
    dag_id="example2",
    description="Example DAG with Operators",
    default_args=default_args,
    schedule_interval=None,
    start_date=days_ago(2),
    tags=["example"],
)
def example_2():
    # Tasks
    task_1 = PythonOperator(task_id="task_1", python_callable=_task_1)
    task_2 = PythonOperator(task_id="task_2", python_callable=_task_2)
    task_1 >> task_2


# Run DAGs
example1_dag = example_1()
example2_dag = example_2()
