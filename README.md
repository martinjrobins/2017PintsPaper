# Pints Matrix

This repo contains the infrastructure neccessary to run the full Pints testing matrix on the Oxford `arcus-b` cluster.

# Getting Started

1. Get an account on `arcus-b`
2. Install your ssh key on this account

```bash
$ ssh-keygen -t rsa
$ ssh-copy-id your_username@arcus-b.arc.ox.ac.uk
```
3. Put the following lines in your `$HOME/.ssh/config` file

```bash
Host arcus-b
    HostName arcus-b.arc.ox.ac.uk
    User your_username
```

# What are the tests

Look in the `main.py` file and look at the variables `models`, `optimisers`, and `noise_levels`

# Running the tests

Run this command in the root directory of this repository

```bash
$ python main.py --execute
```

This will `ssh` into the arcus-b cluster and submit an array job that executes the full matrix of tests.



