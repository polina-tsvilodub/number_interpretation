# Non literal number interpretation by language models

This repo contains materials and data for the paper "Non-literal Understanding of Number Words by Language Models".
The repo is structured as follows:

`analysis`: contains notebooks for analyzing data and producing plots

* `analyse_speaker.ipynb`: notebook for analysing data of LMs predicting likelihoods of utterances, given states and goals (pragmatic speaker simulations) and for exploring data of free generation
* `nesy_listener.ipynb`: notebook for running the fully LLM-based RSA model. Requires LLM speaker data and LLM priors.
* `plot_generation.ipynb`: notebook for wrangling raw LLM data and generating all plots except for the RSA based results.

`prompt_instructions`: contains txt files with all prompts

* `advanced_prompting`: prompts for speaker simulations and for one-shot prompting
* files with `1b`: prompt for experiment 1
* files with `2`: prompt for experiment 2
* files with `3a`: prompt for experiment 3a (price priors)
* files with `3b`: prompt for experiment 3b (affect priors)

`src`: code for generating LLM results (requires an .env file in this directory with the LLM API keys)

* `evaluate_llm_hyperbole.py`: script for generating all scoring based results (exp. 1 -- 3)
* `evaluate_log_p_hyperbole.py`: script for retrieving log probabilities of different prices under an open-source model
* `evaluate_speaker_model.py`: script for eliciting LLM likelihoods of utterances or freely generating utterances, given speaker goals
* `init_model.py`: helper for initializing LLMs within a LangChain wrapper
* `run_slurm_job.sh`: slurm script for running the simulations on a slurm cluster

`data`: contains vignettes formatted for all experiments

`human_data`: contains wrangled human data from Kao et al. (2014)

`results`: contains results combined across runs of the models

* `llms`: results of LLM based runs
* `rsa`: posterior predictions from the RSA model, parameterized with different LLM priors / likelihoods

`rsa.ipynb`: notebook for running the RSA models implemented in WebPPL, running hyperparameter search and plotting results

`rsa.wppl`: probabilistic model of non-literal number understanding from Kao et al. (2014)