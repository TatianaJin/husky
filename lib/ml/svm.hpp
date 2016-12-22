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
#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include "boost/tokenizer.hpp"
#include "core/executor.hpp"
#include "core/objlist.hpp"
#include "lib/aggregator_factory.hpp"
#include "lib/ml/data_loader.hpp"
#include "lib/ml/feature_label.hpp"
#include "lib/ml/model.hpp"
#include "lib/ml/parameter.hpp"

namespace husky {
namespace lib {
namespace ml {

template <typename FeatureT, typename LabelT, bool is_sparse, typename ParamT>
class SVM : public Model<FeatureT, LabelT, is_sparse, ParamT> {
   public:
    using ObjT = husky::lib::ml::LabeledPointHObj<FeatureT, LabelT, is_sparse>;
    using ObjL = ObjList<ObjT>;

    // Constructors
    SVM() : Model<FeatureT, LabelT, is_sparse, ParamT>() {}
    explicit SVM(int _num_features) : Model<FeatureT, LabelT, is_sparse, ParamT>(_num_features + 1) {
        this->num_feature_ = _num_features;

        this->gradient_func_ = [](ObjT& this_obj, Vector<FeatureT, false>& param) {
            double y = this_obj.y;
            auto X = this_obj.x;
            auto prod = param.dot_with_intcpt(X) * y; // prod = WX * y

            if (prod < 1) {  // the data point falls within the margin
                X *= y;
                auto num_param = param.get_feature_num();
                X.resize(num_param);
                X.set(num_param - 1, y);
                return X;
            }
            return Vector<FeatureT, is_sparse>(0);
        };

        this->error_func_ = [](ObjT& this_obj, ParamT& param_list) {
            double indicator = 0;
            auto y = this_obj.y;
            auto X = this_obj.x;
            for (auto it = X.begin_feaval(); it != X.end_feaval(); it++)
                indicator += param_list.param_at((*it).fea) * (*it).val;
            indicator += param_list.param_at(param_list.get_num_param() - 1);  // bias
            indicator *= y;  // right prediction if positive (Wx+b and y have the same sign)
            if (indicator <= 0)
                return 1;
            return 0;
        };
    }

    inline void set_regularization_factor(double _lambda) { lambda_ = _lambda; }

    template <template <typename, typename, bool> typename Optimizer>
    void train(ObjL& data, int num_iters, double learning_rate) {
        ASSERT_MSG(this->param_list_.get_num_param() > 0, "The number of parameters is 0.");

        // get the number of global records
        Aggregator<int> num_samples_agg(0, [](int& a, const int& b) { a += b; });
        num_samples_agg.update(data.get_size());
        AggregatorFactory::sync();
        int num_samples = num_samples_agg.get_value();
        if (Context::get_global_tid() == 0) {
            base::log_info("Training set size = " + std::to_string(num_samples));
        }

        Optimizer<FeatureT, LabelT, is_sparse> optimizer(this->gradient_func_, learning_rate);
        if (Context::get_global_tid() == 0) {
            optimizer.set_regularization(2, lambda_);
        }

        auto& ac = AggregatorFactory::get_channel();
        Aggregator<double> loss_agg(0.0, [](double& a, const double& b) { a += b; });
        loss_agg.to_reset_each_iter();

        auto start = std::chrono::steady_clock::now();
        for (int round = 0; round < num_iters; ++round) {
            optimizer.update_param(data, this->param_list_, num_samples);
            if (this->report_per_round) {
                list_execute(data, {}, {&ac}, [&](ObjT& this_obj) {
                    double y = this_obj.y;
                    auto X = this_obj.x;
                    auto prod = this->param_list_.param_at(this->num_feature_);  // bias
                    for (auto it = X.begin_feaval(); it != X.end_feaval(); it++)
                        prod += this->param_list_.param_at((*it).fea) * (*it).val;
                    if (prod < 1)
                        loss_agg.update(1 - prod);
                });

                if (husky::Context::get_global_tid() == 0) {
                    auto loss = loss_agg.get_value() / num_samples;
                    husky::base::log_info("Iteration " + std::to_string(round + 1) + ": loss = " + std::to_string(loss));
                }
            }
        }
        auto end = std::chrono::steady_clock::now();
        // report training time
        if (husky::Context::get_global_tid() == 0) {
            husky::base::log_info(
                "Time: " + std::to_string(std::chrono::duration_cast<std::chrono::duration<float>>(end - start).count()));
        }
        this->trained_ = true;
    }  // end of train()

   private:
    double lambda_;
};

}  // namespace ml
}  // namespace lib
}  // namespace husky
