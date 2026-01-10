from etna import Model

def test_minibatch_training_runs():
    # Create a tiny CSV file for testing
    csv_content = """x1,x2,label
0.1,0.2,A
0.2,0.3,A
0.3,0.4,B
0.4,0.5,B
"""

    with open("test_data.csv", "w") as f:
        f.write(csv_content)

    model = Model("test_data.csv", target="label")

    # Train using mini-batches
    model.train(epochs=5, lr=0.01, batch_size=2)

    # Assert loss history is populated
    assert len(model.loss_history) > 0
