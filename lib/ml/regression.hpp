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

#pragma once

#include <algorithm>
#include <functional>

#include "core/context.hpp"
#include "core/objlist.hpp"
#include "core/utils.hpp"
#include "lib/aggregator.hpp"
#include "lib/aggregator_factory.hpp"
#include "lib/ml/feature_label.hpp"
#include "lib/ml/model.hpp"
#include "lib/ml/parameter.hpp"
#include "lib/vector.hpp"

namespace husky {
namespace lib {
namespace ml {

using husky::lib::AggregatorFactory;
using husky::lib::Aggregator;

// base class for regression
template <typename FeatureT, typename LabelT, bool is_sparse, typename ParamT>
class Regression : public Model<FeatureT, LabelT, is_sparse, ParamT> {
    typedef LabeledPointHObj<FeatureT, LabelT, is_sparse> ObjT;
    typedef ObjList<ObjT> ObjL;

   public:
    // constructors
    Regression() : Model<FeatureT, LabelT, is_sparse, ParamT>() {}
    explicit Regression(int _num_param) : Model<FeatureT, LabelT, is_sparse, ParamT>(_num_param) {}

    // train model
    template <template <typename, typename, bool> typename GD>
    void train(ObjL& data, int iters, double learning_rate) {
        // check conditions
        ASSERT_MSG(this->param_list_.get_num_param() > 0, "The number of parameters is 0.");
        ASSERT_MSG(this->gradient_func_ != nullptr, "Gradient function is not specified.");
        ASSERT_MSG(this->error_func_ != nullptr, "Error function is not specified.");

        // statistics: total number of samples and error
        Aggregator<int> num_samples_agg(0, [](int& a, const int& b) { a += b; });            // total number of samples
        Aggregator<FeatureT> error_stat(0.0, [](FeatureT& a, const double& b) { a += b; });  // sum of error
        error_stat.to_reset_each_iter();                                                     // reset each round

        // get statistics
        auto& ac = AggregatorFactory::get_channel();
        list_execute(data, {}, {&ac}, [&](ObjT& this_obj) { num_samples_agg.update(1); });
        int num_samples = num_samples_agg.get_value();
        // report statistics
        if (Context::get_global_tid() == 0) {
            husky::base::log_info("Training set size = " + std::to_string(num_samples));
        }

        // use gradient descent to calculate step
        GD<FeatureT, LabelT, is_sparse> gd(this->gradient_func_, learning_rate);

        for (int round = 0; round < iters; round++) {
            // delegate update operation to gd
            gd.update_param(data, this->param_list_, num_samples);

            // option to report error rate
            if (this->report_per_round == true) {
                // calculate error rate
                list_execute(data, {}, {&ac}, [&, this](ObjT& this_obj) {
                    auto error = this->error_func_(this_obj, this->param_list_);
                    error_stat.update(error);
                });
                if (Context::get_global_tid() == 0) {
                    base::log_info("The error in iteration " + std::to_string(round + 1) + ": " +
                                  std::to_string(error_stat.get_value() / num_samples));
                }
            }
        }
        this->trained_ = true;
    }  // end of train

    // train and test model with early stopping
    template <template <typename, typename, bool> typename GD>
    void train_test(ObjL& data, ObjL& Test, int iters, double learning_rate) {
        // check conditions
        ASSERT_MSG(this->param_list_.get_num_param() > 0, "The number of parameters is 0.");
        ASSERT_MSG(this->gradient_func_ != nullptr, "Gradient function is not specified.");
        ASSERT_MSG(this->error_func_ != nullptr, "Error function is not specified.");

        // statistics: total number of samples and error
        Aggregator<int> num_samples_agg(0, [](int& a, const int& b) { a += b; });  // total number of samples
        Aggregator<FeatureT> error_stat(0.0, [](FeatureT& a, const FeatureT& b) { a += b; });  // sum of error
        error_stat.to_reset_each_iter();                                                       // reset each round

        // get statistics
        auto& ac = AggregatorFactory::get_channel();
        list_execute(data, {}, {&ac}, [&](ObjT& this_obj) { num_samples_agg.update(1); });
        int num_samples = num_samples_agg.get_value();
        // report statistics
        if (Context::get_global_tid() == 0) {
            husky::base::log_info("Training set size = " + std::to_string(num_samples));
        }

        // use gradient descent to calculate step
        GD<FeatureT, LabelT, is_sparse> gd(this->gradient_func_, learning_rate);
        double pastError = 0.0;

        for (int round = 0; round < iters; round++) {
            // delegate update operation to gd
            gd.update_param(data, this->param_list_, num_samples);

            auto currentError = avg_error(Test);
            // option to report error rate
            if (this->report_per_round == true) {
                if (Context::get_global_tid() == 0) {
                    base::log_info("The error in iteration " + std::to_string(round + 1) + ": " +
                                  std::to_string(currentError));
                }
            }

            // TODO(Tatiana): handle fluctuation in test error
            // validation based early stopping -- naive version
            if (currentError == 0.0 || (round != 0 && currentError > pastError)) {
                if (Context::get_global_tid() == 0) {
                    base::log_info("Early stopping invoked. Training is completed.");
                }
                break;
            }
            pastError = currentError;
        }

        this->trained_ = true;
    }  // end of train_test
};  // Regression

}  // namespace ml
}  // namespace lib
}  // namespace husky
