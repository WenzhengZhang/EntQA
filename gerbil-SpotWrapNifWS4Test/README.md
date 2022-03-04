Gerbil
========
This directory is copied from [end2end_neural_el](https://github.com/dalab/end2end_neural_el) repository.
<i>General Entity Annotator Benchmark</i>

This branch is part of the Gerbil project. It contains a very simple Webservice that wraps the DBpedia Spotlight Webservice and implements the NIF communication. We use it to test the NIF based Webservice annotator adapter that is part of Gerbil.

# GERBIL Evaluation

Download [GERBIL repository](https://github.com/dice-group/gerbil). On one terminal run Gerbil. Execute:

```
cd gerbil                     
./start.sh
```
Caution: Gerbil might be incompatible with some java versions. With Java 8 it works.

On another terminal execute:

```
cd gerbil-SpotWrapNifWS4Test
mvn clean -Dmaven.tomcat.port=1235 tomcat:run
```

On the third terminal execute to run our model:

```
python server.py
```

Open ```http://IP_address/gerbil/config```, select A2KB as Experiment type, strong/weak evaluation as Matching, fill out any random name and ```http://ID_address:1235/gerbil-spotWrapNifWS4Test/myalgorithm``` for URI. Select datasets to run on and click Run Experiment.

The experiment will be running properly.
