# eoai_challenge
## To run
`docker build -t eoai_task .`

`docker run -p 8888:8888 -v local_path_to_tlg_eoai_task_results_folder:/app/notebooks -v local_path_to_tlg_eoai_task_results_folder/data:/app/data eoai_test`

go to `http://localhost:8888` to access the notebook.
