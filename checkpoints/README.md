# Model Checkpoints

Models checkpoints will be made available for each experiment. The checkpoints will be available in the following format:

```
checkpoints/
├── dataset
│   ├── num_clients
│   │   ├── run_description.txt
│   │   ├── server
│   │   │   ├── model.pt
│   │   │   ├── optimizer.pt
│   │   ├── client_0
│   │   │   ├── model.pt
│   │   │   ├── optimizer.pt
│   │   ├── client_1
│   │   │   ├── model.pt
│   │   │   ├── optimizer.pt
│   │   └── ...
│   └── ...
└── ...
```