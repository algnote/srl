import math, os
import numpy as np
import paddle.v2 as paddle
import paddle.v2.evaluator as evaluator
import pickle

with_gpu = os.getenv('WITH_GPU', '0') != '0'

print("loding data...")
f = open('data/train.data', 'rb')
data = pickle.load(f)
print("done")

for i in range(len(data)):
    data[i] = map(list, zip(*data[i]))

mark_dict_len = 31
pos_dict_len = 32
label_dict_len = 68
word_dim = 64
mark_dim = 5
hidden_dim = 512
depth = 8
default_std = 1 / math.sqrt(hidden_dim) / 3.0
mix_hidden_lr = 1e-3


def d_type(size):
    return paddle.data_type.dense_vector_sequence(size)

def b_type(size):
    return paddle.data_type.sparse_binary_vector_sequence(size)

def i_type(size):
    return paddle.data_type.integer_value_sequence(size)


def db_lstm():
    #14 features
    word = paddle.layer.data(name='word_data', type=d_type(word_dim))
    predicate = paddle.layer.data(name='verb_data', type=d_type(word_dim))

    ctx_n2 = paddle.layer.data(name='ctx_n2_data', type=d_type(word_dim))
    ctx_n1 = paddle.layer.data(name='ctx_n1_data', type=d_type(word_dim))
    ctx_0 = paddle.layer.data(name='ctx_0_data', type=d_type(word_dim))
    ctx_p1 = paddle.layer.data(name='ctx_p1_data', type=d_type(word_dim))
    ctx_p2 = paddle.layer.data(name='ctx_p2_data', type=d_type(word_dim))
    ptx_n2 = paddle.layer.data(name='ptx_n2_data', type=b_type(pos_dict_len))
    ptx_n1 = paddle.layer.data(name='ptx_n1_data', type=b_type(pos_dict_len))
    ptx_0 = paddle.layer.data(name='ptx_0_data', type=b_type(pos_dict_len))
    ptx_p1 = paddle.layer.data(name='ptx_p1_data', type=b_type(pos_dict_len))
    ptx_p2 = paddle.layer.data(name='ptx_p2_data', type=b_type(pos_dict_len))
    mark = paddle.layer.data(name='mark_data', type=b_type(mark_dict_len))
    pos = paddle.layer.data(name='pos_data', type=b_type(pos_dict_len))


    std_0 = paddle.attr.Param(initial_std=0.)
    std_default = paddle.attr.Param(initial_std=default_std)

    data_layers = [word, ctx_n2, ctx_n1, ctx_0, ctx_p1, ctx_p2, ptx_n2, ptx_n1, ptx_0, ptx_p1, ptx_p2, predicate, mark, pos]

    hidden_0 = paddle.layer.mixed(
        size=hidden_dim,
        bias_attr=std_default,
        input=[
            paddle.layer.full_matrix_projection(
                input=d, param_attr=std_default) for d in data_layers
        ])

    lstm_para_attr = paddle.attr.Param(initial_std=0.0, learning_rate=1.0)
    hidden_para_attr = paddle.attr.Param(
        initial_std=default_std, learning_rate=mix_hidden_lr)

    lstm_0 = paddle.layer.lstmemory(
        input=hidden_0,
        act=paddle.activation.Relu(),
        gate_act=paddle.activation.Sigmoid(),
        state_act=paddle.activation.Sigmoid(),
        bias_attr=std_0,
        param_attr=lstm_para_attr)

    #stack L-LSTM and R-LSTM with direct edges
    input_tmp = [hidden_0, lstm_0]

    for i in range(1, depth):
        mix_hidden = paddle.layer.mixed(
            size=hidden_dim,
            bias_attr=std_default,
            input=[
                paddle.layer.full_matrix_projection(
                    input=input_tmp[0], param_attr=hidden_para_attr),
                paddle.layer.full_matrix_projection(
                    input=input_tmp[1], param_attr=lstm_para_attr)
            ])

        lstm = paddle.layer.lstmemory(
            input=mix_hidden,
            act=paddle.activation.Relu(),
            gate_act=paddle.activation.Sigmoid(),
            state_act=paddle.activation.Sigmoid(),
            reverse=((i % 2) == 1),
            bias_attr=std_0,
            param_attr=lstm_para_attr)

        input_tmp = [mix_hidden, lstm]

    feature_out = paddle.layer.mixed(
        size=label_dict_len,
        bias_attr=std_default,
        input=[
            paddle.layer.full_matrix_projection(
                input=input_tmp[0], param_attr=hidden_para_attr),
            paddle.layer.full_matrix_projection(
                input=input_tmp[1], param_attr=lstm_para_attr)
        ], )

    return feature_out


def load_parameter(file_name, h, w):
    with open(file_name, 'rb') as f:
        f.read(16)  # skip header.
        return np.fromfile(f, dtype=np.float32).reshape(h, w)


def main():
    paddle.init(use_gpu=with_gpu, trainer_count=1)

    # define network topology
    feature_out = db_lstm()
    target = paddle.layer.data(name='target', type=i_type(label_dict_len))
    

    # CRF State
    crf_cost = paddle.layer.crf(
        size=label_dict_len,
        input=feature_out,
        label=target,
        param_attr=paddle.attr.Param(
            name='crfw', initial_std=default_std, learning_rate=mix_hidden_lr))
    
    # CRF Transfer 
    crf_dec = paddle.layer.crf_decoding(
        size=label_dict_len,
        input=feature_out,
        label=target,
        param_attr=paddle.attr.Param(name='crfw'))
    evaluator.sum(input=crf_dec)

    # create parameters
    parameters = paddle.parameters.create(crf_cost)

    # create optimizer
    optimizer = paddle.optimizer.Momentum(
        momentum=0,
        learning_rate=2e-2,
        regularization=paddle.optimizer.L2Regularization(rate=8e-4),
        model_average=paddle.optimizer.ModelAverage(
            average_window=0.5, max_average_window=10000), )

    trainer = paddle.trainer.SGD(
        cost=crf_cost,
        parameters=parameters,
        update_equation=optimizer,
        extra_layers=crf_dec)

    train_data = data[:17840]
    dev_data = data[17840:18956]
    test_data = data[18956:]
    def reader1():
        for i in xrange(len(train_data)):
            yield  train_data[i]
    def dev_reader1():
        for i in xrange(len(dev_data)):
            yield  dev_data[i]
    
    reader = paddle.batch(
        paddle.reader.shuffle(
            reader1, buf_size=8192), batch_size=10)

    dev_reader = paddle.batch(
        paddle.reader.shuffle(
            dev_reader1, buf_size=8192), batch_size=10)


    feeding = {
        'word_data': 0,
        'verb_data': 1,
        'ctx_n2_data': 2,
        'ctx_n1_data': 3,
        'ctx_0_data': 4,
        'ctx_p1_data': 5,
        'ctx_p2_data': 6,
        'ptx_n2_data': 7,
        'ptx_n1_data': 8,
        'ptx_0_data': 9,
        'ptx_p1_data': 10,
        'ptx_p2_data': 11,
        'mark_data': 12,
        'pos_data': 13,
        'target': 14
    }

    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 100 == 0:
                print "Pass %d, Batch %d, Cost %f, %s" % (
                    event.pass_id, event.batch_id, event.cost, event.metrics)
            if event.batch_id % 1000 == 0:
                result = trainer.test(reader=dev_reader, feeding=feeding)
                print "\nTest with Pass %d, Batch %d, %s" % (
                    event.pass_id, event.batch_id, result.metrics)

        if isinstance(event, paddle.event.EndPass):
            # save parameters
            with open('params_pass_%d.tar' % event.pass_id, 'w') as f:
                trainer.save_parameter_to_tar(f)

            result = trainer.test(reader=dev_reader, feeding=feeding)
            print "\nTest with Pass %d, %s" % (event.pass_id, result.metrics)

    trainer.train(
        reader=reader,
        event_handler=event_handler,
        num_passes=2,
        feeding=feeding)

    dev_data = map(lambda x: x[:14], dev_data)
    test_data = map(lambda x: x[:14], test_data)

    predict = paddle.layer.crf_decoding(
        size=label_dict_len,
        input=feature_out,
        param_attr=paddle.attr.Param(name='crfw'))
    
    probs1 = paddle.infer(
        output_layer=predict,
        parameters=parameters,
        input=dev_data,
        feeding=feeding,
        field='id')
    
    probs2 = paddle.infer(
        output_layer=predict,
        parameters=parameters,
        input=test_data,
        feeding=feeding,
        field='id')

    f = open('data/dev.pre', 'wb')
    pickle.dump(probs1, f)
    f.close()

    f = open('data/test.pre', 'wb')
    pickle.dump(probs2, f)
    f.close()


if __name__ == '__main__':
    main()