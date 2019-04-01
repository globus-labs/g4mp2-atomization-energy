from jcesr_ml.benchmark import load_benchmark_data


def test_train_unchanged():
    """Make sure the training entries have not changed"""

    train_data, test_data = load_benchmark_data()
    assert len(train_data) == 117232
    assert hash(tuple(train_data['index'])) == 364816341895120010

    assert len(test_data) == 13026
    assert hash(tuple(test_data['index'])) == 4595558060980866953
