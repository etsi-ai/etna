# etsi.etna

> **What if machine learning felt effortless?**

`etsi.etna` is a minimalistic, dependency-light neural network library
designed to make training and evaluating models on structured data fast,
interpretable, and beginner-friendly. It focuses on auto-preprocessing,
simple linear networks, and core metrics --- ideal for research
prototyping, learning, and quick deployments.

------------------------------------------------------------------------

## 🚀 Quickstart

```bash
 pip install etsi-etna
```

```bash 
import etsi.etna as etna

model = etna.Model("diabetes.csv", target="Outcome") model.train()
model.evaluate()
```
------------------------------------------------------------------------

## 🔮 Features

📦 One-liner dataset ingestion (.csv, .txt)

🧼 Automatic preprocessing (scaling, encoding)

🧠 Core NN: Linear → ReLU → Softmax

📊 Built-in evaluation (accuracy, F1)

🔍 CLI support

🪶 No hard dependencies --- minimal & fast

## 🤝 Contributing Pull requests are welcome!

Please refer to [CONTRIBUTING.md](https://github.com/etsi-ai/etna/blob/main/CONTRIBUTING.md) for contribution guidelines and ensure
your code passes:

bash Copy Edit make check Use 4 spaces for indentation

Follow consistent commit messages Example: \[fix\] model.py: fixed
one-hot bug

By contributing, you agree to license your code under the BSD-2-Clause
license.

## 📄 License This project is distributed under the BSD-2-Clause License.

Built with ❤️ by etsi.ai "Making machine learning simple, again."


------------------------------------------------------------------------

