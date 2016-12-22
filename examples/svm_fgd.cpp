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
// penalty
// type: double
// info: penalty factor for hinge loss
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
// configuration example:
// train=hdfs:///datasets/classification/a9
// test=hdfs:///datasets/classification/a9t
// is_sparse=true
// format=libsvm
// n_iter=100
// penalty=100

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
    double C = std::stod(husky::Context::get_param("penalty"));
    int num_iter = std::stoi(husky::Context::get_param("n_iter"));

    // initialize parameters
    husky::lib::ml::ParameterBucket<double> param_list(num_features + 1);  // scalar b and vector w

    if (husky::Context::get_global_tid() == 0) {
        husky::base::log_info("num of params: " + std::to_string(param_list.get_num_param()));
    }

    // get the number of global records
    Aggregator<int> num_samples_agg(0, [](int& a, const int& b) { a += b; });
    num_samples_agg.update(train_set.get_size());
    AggregatorFactory::sync();
    int num_samples = num_samples_agg.get_value();
    if (husky::Context::get_global_tid() == 0) {
        husky::base::log_info("Training set size = " + std::to_string(num_samples));
    }

    // Aggregators for regulator, w square and loss
    Aggregator<double> loss_agg(0.0, [](double& a, const double& b) { a += b; });
    loss_agg.to_reset_each_iter();

    // Main loop
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < num_iter; i++) {
        // get local copy of parameters
        auto bweight = param_list.get_all_param();

        // decaying learning rate
        double eta = 1. / (i + 1);

        // regularize w in param_list
        if (husky::Context::get_global_tid() == 0) {
            for (int idx = 0; idx < num_features; idx++) {
                double w = bweight[idx];
                param_list.update(idx, - eta * w);
            }
        }

        auto& ac = AggregatorFactory::get_channel();
        // calculate gradient
        list_execute(train_set, {}, {&ac}, [&](ObjT& this_obj) {
            double y = this_obj.y;
            auto X = this_obj.x;
            auto prod = bweight.dot_with_intcpt(X) * y; // prod = WX * y

            if (prod < 1) {  // the data point falls within the margin
                for (auto it = X.begin_feaval(); it != X.end_feaval(); ++it) {
                    auto x = *it;
                    x.val *= y;  // calculate the gradient for each parameter
                    param_list.update(x.fea, eta * x.val / num_samples * C);
                }
                // update bias
                param_list.update(num_features, eta * y / num_samples * C);
                loss_agg.update(1 - prod);
            }
        });

        int num_samples = num_samples_agg.get_value();
        double loss = loss_agg.get_value() / num_samples;
        if (husky::Context::get_global_tid() == 0) {
            husky::base::log_info("Iteration " + std::to_string(i + 1) + ": loss = " + std::to_string(loss));
        }
    }
    auto end = std::chrono::steady_clock::now();

    // Show result
    if (husky::Context::get_global_tid() == 0) {
        param_list.present();
        husky::base::log_info(
            "Time: " + std::to_string(std::chrono::duration_cast<std::chrono::duration<float>>(end - start).count()));
    }

    // test
    Aggregator<int> error_agg(0, [](int& a, const int& b) { a += b; });
    Aggregator<int> num_test_agg(0, [](int& a, const int& b) { a += b; });
    auto& ac = AggregatorFactory::get_channel();
    auto bweight = param_list.get_all_param();
    list_execute(test_set, {}, {&ac}, [&](ObjT& this_obj) {
        double indicator = 0;
        auto y = this_obj.y;
        auto X = this_obj.x;
        for (auto it = X.begin_feaval(); it != X.end_feaval(); it++)
            indicator += bweight[(*it).fea] * (*it).val;
        // bias
        indicator += bweight[num_features];
        indicator *= y;  // right prediction if positive (Wx+b and y have the same sign)
        if (indicator < 0)
            error_agg.update(1);
        num_test_agg.update(1);
    });

    if (husky::Context::get_global_tid() == 0) {
        husky::base::log_info("Error rate on testing set: " +
                             std::to_string(static_cast<double>(error_agg.get_value()) / num_test_agg.get_value()));
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
    std::vector<std::string> args(
        {"hdfs_namenode", "hdfs_namenode_port", "train", "test", "n_iter", "penalty", "format", "is_sparse"});
    if (husky::init_with_args(argc, argv, args)) {
        husky::run_job(init);
        return 0;
    }
    return 1;
}
