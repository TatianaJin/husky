// Copyright 2016 Husky Team
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//
// Parameters
//
// lambda
// type: double
// info: regularization parameter
//
// train
// type: string
// info: the path of training data in hadoop, in LibSVM format
//
// test
// type: string
// info: the path of testing data in hadoop, in LibSVM format
//
// n_iter
// type: int
// info: number of epochs the entire training data will be went through
//
// is_sparse
// type: string
// info: whether the data is dense or sparse
//
// format
// type: string
// info: the data format of input file: libsvm/tsv
//
// alpha
// type: double
// info: the learning rate
//
// configuration example:
// train=hdfs:///datasets/classification/a9
// test=hdfs:///datasets/classification/a9t
// is_sparse=true
// format=libsvm
// n_iter=50
// lambda=0.01
// alpha=0.01

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include "boost/tokenizer.hpp"

#include "core/engine.hpp"
#include "lib/ml/data_loader.hpp"
#include "lib/ml/feature_label.hpp"
#include "lib/ml/parameter.hpp"
#include "lib/ml/sgd.hpp"
#include "lib/ml/svm.hpp"

using husky::lib::Aggregator;
using husky::lib::AggregatorFactory;

template <bool is_sparse>
void svm() {
    using ObjT = husky::lib::ml::LabeledPointHObj<double, double, is_sparse>;
    auto& train_set = husky::ObjListStore::create_objlist<ObjT>("train_set");
    auto& test_set = husky::ObjListStore::create_objlist<ObjT>("test_set");

    auto format_str = husky::Context::get_param("format");
    husky::lib::ml::DataFormat format;
    if (format_str == "libsvm") {
        format = husky::lib::ml::kLIBSVMFormat;
    } else if (format_str == "tsv") {
        format = husky::lib::ml::kTSVFormat;
    }

    // load data
    int num_features = husky::lib::ml::load_data(husky::Context::get_param("train"), train_set, format);
    husky::lib::ml::load_data(husky::Context::get_param("test"), test_set, format);

    // get model config parameters
    double lambda = std::stod(husky::Context::get_param("lambda"));
    double alpha = std::stod(husky::Context::get_param("alpha"));
    int num_iter = std::stoi(husky::Context::get_param("n_iter"));

    husky::lib::ml::SVM<double, double, is_sparse, husky::lib::ml::ParameterBucket<double>> model(num_features);
    model.report_per_round = true;
    model.set_regularization_factor(lambda);

    model.template train<husky::lib::ml::SGD>(train_set, num_iter, alpha);
    if (husky::Context::get_global_tid() == 0)
        model.present_param();

    auto test_error = model.avg_error(test_set);
    if (husky::Context::get_global_tid() == 0) {
        husky::base::log_info("The error rate on testing set = " + std::to_string(test_error));
    }
}

void init() {
    if (husky::Context::get_param("is_sparse") == "true") {
        svm<true>();
    } else {
        svm<false>();
    }
}

int main(int argc, char** argv) {
    std::vector<std::string> args({
        "hdfs_namenode", "hdfs_namenode_port", "train", "test", "n_iter", "lambda", "format", "is_sparse", "alpha"});
    if (husky::init_with_args(argc, argv, args)) {
        husky::run_job(init);
        return 0;
    }
    return 1;
}
