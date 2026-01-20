
# MYAMR TGDK
This is the companion code for the contributions reported in the paper LLM-Supported Manufacturing Mapping Generation by Wilma Johanna Schmidt et al. 
The paper will be published in the journal Transactions on Graph Data and Knowledge (TGDK).
The code allows the users to reproduce and extend the results reported in the paper. 
Please cite the above paper when reporting, reproducing or extending the results.

## Purpose of the project

This software is a research prototype, solely developed for and published as part of the publication LLM-Supported Manufacturing Mapping Generation. 
It will neither be maintained nor monitored in any way.

The following folder structure is expected for myamr_tgdk to run smoothly:

```xml
mapping-generation
 |-- input
 |    |-- generic
 |    |    |-- prompt_rml
 |    |    |    |-- data.csv
 |    |    |    |-- few_shot.json
 |    |    |    |-- ontology.ttl
 |    |    |-- prompt_yarrrml
 |    |    |    |-- data.csv
 |    |    |    |-- few_shot.json
 |    |    |    |-- ontology.ttl
 |    |    |-- reference_rml.ttl
 |    |    |-- reference_yarrrml.yml
 |    |-- task instructions
 |    |    |-- rml
 |    |    |    |-- 1-few_shot.txt
 |    |    |    |-- template.txt
 |    |    |-- yarrrml
 |    |    |    |-- 0_shot.txt 
 |    |    |    |-- 1-few_shot.txt
 |    |    |    |-- template.txt
 |-- generated_files
 |    |-- generic
 |-- .env
 |-- eval_rml_relaxed.py
 |-- eval_yarrrml_relaxed.py
 |-- mapping_generation.py
 |-- onto_reducer_llm.py
 |-- onto_reducer_naive.py
 |-- onto_reducer_similar.py

```
To install the environment needed to run myamr_tgdk, make sure you have a conda environment with Python (tested on 3.13.3) running and install packages listed below:

````bash
$ pip install rdflib (tested on 7.1.4)
$ pip install langchain-openai (tested on 0.3.17)
$ pip install openai (tested on 1.78.1)
$ pip install python-dotenv (tested on 1.1.0)
$ pip install numpy (tested on 2.2.5)
$ pip install sentence-transformers (tested on 4.1.0)
$ pip install scikit-learn (tested on 1.6.1)
$ pip install ruamel-yaml (tested on 0.18.10)
$ pip install pandas (tested on 2.2.3)
$ pip install langchain (tested on 0.3.25)
$ pip install langchain-community (tested on 0.3.24)
$ pip install azure-identity (tested on 1.23.0)
$ pip install faiss-cpu (tested on 1.11.0)
````

Depending on Azure or other LLM APIs, you might need to install further the packages.


License Information

myamr_tgdk is open-sourced with the following license: the code part is licensed under the AGPL-3.0 license.

See the license files LICENSE_AGPL-3.0.txt for details.
