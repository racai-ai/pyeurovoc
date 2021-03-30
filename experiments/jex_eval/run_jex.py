import os
from random import shuffle
import random
import shutil


jex_path = "eurovoc"
config_dir = "config_splits"
classifiers_dir = "classifiers_splits"
results_dir = "results"
SEEDS = [110, 221]


def create_splits(directory):
    if not os.path.exists(os.path.join(jex_path, directory, "workspace", "splits")):
        os.makedirs(os.path.join(jex_path, directory, "workspace", "splits"))
    else:
        shutil.rmtree(os.path.join(jex_path, directory, "workspace", "splits"))
        os.makedirs(os.path.join(jex_path, directory, "workspace", "splits"))

    list_samples = []

    with open(os.path.join(jex_path, directory, "workspace", "corpus.cf"), "r", encoding="utf-8") as file:
        for labels in file:
            document = file.readline()

            list_samples.append((document, labels))

    for i, seed in enumerate(SEEDS):
        random.seed(seed)
        shuffle(list_samples)

        ratio = int(len(list_samples) * 0.9)
        train = list_samples[:ratio]
        test = list_samples[ratio:]

        if not os.path.exists(os.path.join(jex_path, directory, "workspace", "splits", "split_train_{}".format(i))):
            os.makedirs(os.path.join(jex_path, directory, "workspace", "splits", "split_train_{}".format(i)))

        if not os.path.exists(os.path.join(jex_path, directory, "workspace", "splits", "split_test_{}".format(i))):
            os.makedirs(os.path.join(jex_path, directory, "workspace", "splits", "split_test_{}".format(i)))

        with open(os.path.join(jex_path, directory, "workspace", "splits", "split_train_{}".format(i), "train.cf"), "w", encoding="utf-8") as file:
            for document, label in train:
                file.write(label)
                file.write(document)

        with open(os.path.join(jex_path, directory, "workspace", "splits", "split_test_{}".format(i), "test.cf"), "w", encoding="utf-8") as file:
            for document, label in test:
                file.write(label)
                file.write(document)


def create_configs(directory):
    if not os.path.exists(os.path.join(jex_path, directory, "workspace", config_dir)):
        os.makedirs(os.path.join(jex_path, directory, "workspace", config_dir))

    for i in range(len(SEEDS)):
        with open(os.path.join(jex_path, directory, "workspace", config_dir, "PreProcess.properties".format(i)), "w") as file:
            file.write("inputDir = {}/{}/workspace/cf\n".format(jex_path, directory))
            file.write("stopwords =  {}/{}/resources/stopwords.txt\n".format(jex_path, directory))
            file.write("output = {}/{}/workspace/corpus.cf\n".format(jex_path, directory, i))
            file.write("forTraining = true\n")
            file.write("MultiWordsFile = {}/{}/resources/stopwords_multi.txt\n".format(jex_path, directory))

        if not os.path.exists(os.path.join(jex_path, directory, "workspace", config_dir, "split_{}".format(i))):
            os.makedirs(os.path.join(jex_path, directory, "workspace", config_dir, "split_{}".format(i)))

        with open(os.path.join(jex_path, directory, "workspace", config_dir, "split_{}".format(i), "Train.properties"), "w") as file:
            file.write("input = {}/{}/workspace/splits/split_train_{}/train.cf\n".format(jex_path, directory, i))
            file.write("classifiersDir = {}/{}/workspace/{}/classifiers_{}/\n".format(jex_path, directory, classifiers_dir, i))
            file.write("dict = {}/{}/workspace/dict_{}\n".format(jex_path, directory, i))
            file.write("dumpReadable = true\n")
            file.write("RefCorpusWordMinFreq = 4\n")
            file.write("LlhThreshold = 5\n")
            file.write("beta = 10\n")
            file.write("minDocLength = 100\n")
            file.write("minNumTrainingDocsPerDescriptor = 4\n")
            file.write("minWeightOfAssociates = 2\n")
            file.write("minNumAssociatesPerDescriptor = 1\n")
            file.write("maxNumAssociatesPerDescriptor = 5000\n")
            file.write("ThesaurusInfo = {}/{}/resources/ThesaurusStructure/\n".format(jex_path, directory))
            file.write("DeprecatedFile = {}/{}/workspace/deprecated_{}.txt\n".format(jex_path, directory, i))

        with open(os.path.join(jex_path, directory, "workspace", config_dir, "split_{}".format(i), "Evaluate.properties"), "w") as file:
            file.write("input = {}/{}/workspace/splits/split_test_{}/test.cf\n".format(jex_path, directory, i))
            file.write("output = {}/{}/workspace/{}/evaluate-result_{}.txt\n".format(jex_path, directory, results_dir, i))
            file.write("dict = {}/{}/workspace/dict_{}\n".format(jex_path, directory, i))
            file.write("blacklist = {}/{}/resources/blacklist.txt\n".format(jex_path, directory))
            file.write("classifiersDir = {}/{}/workspace/{}/classifiers_{}/\n".format(jex_path, directory, classifiers_dir, i))
            file.write("minNumCommonTokens = 1\n")
            file.write("rank = 6\n")
            file.write("displayClassifierInfo = true\n")

        with open(os.path.join(jex_path, directory, "workspace", config_dir, "split_{}".format(i), "Index.properties"), "w") as file:
            file.write("input = {}/{}/workspace/splits/split_test_{}/test.cf\n".format(jex_path, directory, i))
            file.write("output = {}/{}/workspace/{}/Assign_Result_{}.xml\n".format(jex_path, directory, results_dir, i))
            file.write("dict = {}/{}/workspace/dict_{}\n".format(jex_path, directory, i))
            file.write("blacklist = {}/{}/resources/blacklist.txt\n".format(jex_path, directory))
            file.write("classifiersDir = {}/{}/workspace/{}/classifiers_{}/\n".format(jex_path, directory, classifiers_dir, i))
            file.write("minNumDesc = 1\n")
            file.write("rank = 6\n")
            file.write("minNumCommonTokens = 1\n")


def preprocess_files(directory):
    os.system("java -Dlogback.configurationFile={} -Xms1400M -jar {} {}".format(
        os.path.join(jex_path, directory, "workspace", "config", "logback.xml"),
        os.path.join(jex_path, directory, "lib", "CreateCompactFormat.jar"),
        os.path.join(jex_path, directory, "workspace", config_dir, "PreProcess.properties"),
    ))


def train_models(directory):
    for i in range(len(SEEDS)):
        if not os.path.exists(os.path.join(jex_path, directory, "workspace", classifiers_dir, "classifiers_{}".format(i))):
            os.makedirs(os.path.join(jex_path, directory, "workspace", classifiers_dir, "classifiers_{}".format(i)))

        os.system("java -Dlogback.configurationFile={} -Xms1400M -jar {} {}".format(
            os.path.join(jex_path, directory, "workspace", "config", "logback.xml"),
            os.path.join(jex_path, directory, "lib", "EuroVocIndexer.jar"),
            os.path.join(jex_path, directory, "workspace", config_dir, "split_{}".format(i), "Train.properties"),
        ))


def evaluate_models(directory):
    if not os.path.exists(os.path.join(jex_path, directory, "workspace", results_dir)):
        os.makedirs(os.path.join(jex_path, directory, "workspace", results_dir))

    for i in range(len(SEEDS)):
        os.system("java -Dlogback.configurationFile={} -Xms1400M -jar {} {}".format(
            os.path.join(jex_path, directory, "workspace", "config", "logback.xml"),
            os.path.join(jex_path, directory, "lib", "EuroVocIndexer.jar"),
            os.path.join(jex_path, directory, "workspace", config_dir, "split_{}".format(i), "Evaluate.properties"),
        ))


def index_test(directory):
    if not os.path.exists(os.path.join(jex_path, directory, "workspace", results_dir)):
        os.makedirs(os.path.join(jex_path, directory, "workspace", results_dir))

    for i in range(len(SEEDS)):
        os.system("java -Dlogback.configurationFile={} -Xms1400M -jar {} {}".format(
            os.path.join(jex_path, directory, "workspace", "config", "logback.xml"),
            os.path.join(jex_path, directory, "lib", "EuroVocIndexer.jar"),
            os.path.join(jex_path, directory, "workspace", config_dir, "split_{}".format(i), "Index.properties"),
        ))


def run_jex():
    for directory in os.listdir(jex_path):
        create_configs(directory)

        preprocess_files(directory)

        create_splits(directory)

        train_models(directory)

        evaluate_models(directory)

        index_test(directory)


if __name__ == '__main__':

    run_jex()



