# All-Pairs-Similarity
Assignment III LwMD

Required libraries:
beir
pickle
numpy
scipy
sklearn
pandas
pyspark
findspark
nltk
tqdm
loguru
spacy
seaborn
matplotlib

Install Spark:
pip install pyspark
or
download Spark from the [official Apache website](https://spark.apache.org/downloads.html)
  - Download the `.tar.gz` and extract it at some directory `/DIR`
  - rename the main folder to `/spark`, so that the spark installation path becomes `DIR/spark`.
- Set `$SPARK_HOME` env variable in `$HOME/./bashrc` by adding the below lines at the end of the `.bashrc` file.

```bash
# https://sparkbyexamples.com/spark/spark-installation-on-linux-ubuntu/
export SPARK_HOME=<PATH-TO-SPARK-INSTALL>
export PATH=$PATH:$SPARK_HOME/bin
```

- Create `$SPARK_HOME/conf/spark-env.sh` containing the following lines:

```bash
export PYTHONPATH=<PROJECT-CONDA-ENV-PATH>/bin/python
export PYSPARK_PYTHON=<PROJECT-CONDA-ENV-PATH>/bin/python
export PYSPARK_DRIVER_PYTHON=<PROJECT-CONDA-ENV-PATH>/bin/python
export SPARK_MASTER_HOST=127.0.0.1
export SPARK_LOCAL_IP=127.0.0.1
export SPARK_WORKER_CORES=<MAX-CPU-CORES>
export SPARK_WORKER_MEMORY=<MAX-RAM-GB>g   # for example, 64g, 32g, 16g, etc.
```

Additional information regarding Spark:
- Make sure to add local source files as dependencies of the SparkContext; they must be sent to worker nodes, which are not in the drivers' (this) working directory.
    - Make sure to **compress and add the whole source folder** (e.g. `src/`) **as a `.zip` archive**, so that the package structure is preserved.
    - Example:
# Add local dependencies (local python source files) to SparkContext and sys.path
src_zip_path = os.path.abspath("../../src.zip")
spark.sparkContext.addPyFile(src_zip_path)
sys.path.insert(0, SparkFiles.getRootDirectory())

- Start the master and worker nodes by running `bash $SPARK_HOME/sbin/start-all.sh`
- Stop the master and worker nodes by running `bash $SPARK_HOME/sbin/stop-all.sh`