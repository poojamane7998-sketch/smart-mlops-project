from src.data import split_dataset


def test_split_has_rows_and_matching_lengths():

    X_train, X_test, y_train, y_test = split_dataset()

    # Training dataset must contain records
    assert len(X_train) > 0

    # Testing dataset must contain records
    assert len(X_test) > 0

    # Feature and target lengths must match
    assert len(X_train) == len(y_train)

    assert len(X_test) == len(y_test)

    # Training and testing columns must match
    assert list(X_train.columns) == list(X_test.columns)