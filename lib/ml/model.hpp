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

#include "core/objlist.hpp"
#include "lib/aggregator.hpp"
#include "lib/aggregator_factory.hpp"
#include "lib/ml/feature_label.hpp"
#include "lib/ml/parameter.hpp"
#include "lib/vector.hpp"

namespace husky {
namespace lib {
namespace ml {

// base class for machine learning models
template <typename FeatureT, typename LabelT, bool is_sparse, typename ParamT = ParameterBucket<double>>
class Model {
    typedef LabeledPointHObj<FeatureT, LabelT, is_sparse> ObjT;
    typedef ObjList<ObjT> ObjL;

   public:
    // constructors
    Model() {}
    explicit Model(int _num_param) { set_num_param(_num_param); }
    Model(
        std::function<Vector<FeatureT, is_sparse>(ObjT&, Vector<FeatureT, false>&)> _gradient_func,  // gradient func
        std::function<FeatureT(ObjT&, ParamT&)> _error_func,                                         // error function
        int _num_param)                                                                              // number of params
        : gradient_func_(_gradient_func),
          error_func_(_error_func) {
        set_num_param(_num_param);
    }

    // initialize parameter with positive size
    void set_num_param(int _num_param) {
        if (_num_param > 0) {
            param_list_.init(_num_param, 0.0);
        }
    }

    // query parameters
    int get_num_param() { return param_list_.get_num_param(); }  // get parameter size
    void present_param() {                                       // print each parameter to log
        if (this->trained_ == true)
            param_list_.present();
    }

    // predict and store in y
    void set_predict_func(std::function<LabelT(ObjT&, ParamT&)> _predict_func) { this->predict_func_ = _predict_func; }
    void predict(ObjL& data) {
        ASSERT_MSG(this->predict_func_ != nullptr, "Predict function is not specified.");
        list_execute(data, [&, this](ObjT& this_obj) {
            auto& y = this_obj.y;
            y = this->predict_func_(this_obj, this->param_list_);
        });
    }

    // calculate average error rate
    void set_error_func(std::function<FeatureT(ObjT&, ParamT&)> _error_func) { this->error_func_ = _error_func; }
    FeatureT avg_error(ObjL& data) {
        Aggregator<int> num_samples_agg(0, [](int& a, const int& b) { a += b; });
        Aggregator<FeatureT> error_agg(0.0, [](FeatureT& a, const FeatureT& b) { a += b; });
        auto& ac = AggregatorFactory::get_channel();
        list_execute(data, {}, {&ac}, [&, this](ObjT& this_obj) {
            error_agg.update(this->error_func_(this_obj, this->param_list_));
            num_samples_agg.update(1);
        });
        int num_samples = num_samples_agg.get_value();
        auto global_error = error_agg.get_value();
        auto mean_error = global_error / num_samples;
        return mean_error;
    }

    void set_gradient_func(std::function<Vector<FeatureT, is_sparse>(ObjT&, Vector<FeatureT, false>&)> _gradient_func) {
        this->gradient_func_ = _gradient_func;
    }
    bool report_per_round = false;  // whether report error per iteration

   protected:
    std::function<Vector<FeatureT, is_sparse>(ObjT&, Vector<FeatureT, false>&)> gradient_func_ = nullptr;
    std::function<FeatureT(ObjT&, ParamT&)> error_func_ = nullptr;  // error function
    std::function<LabelT(ObjT&, ParamT&)> predict_func_ = nullptr;
    ParamT param_list_;     // parameter vector list
    int num_feature_ = -1;  // number of features (maybe != num_param)
    bool trained_ = false;  // indicate if model is trained
};                          // Model

}  // namespace ml
}  // namespace lib
}  // namespace husky
