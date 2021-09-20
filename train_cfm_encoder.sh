mkdir -p out
mkdir -p out/cfm
mkdir -p out/cfm/chopsticks
mkdir -p out/cfm/rope
mkdir -p out/cfm/torus
mkdir -p out/cfm/writer

python -m plb.algorithms.cfm.train_cfm -d chopsticks -b 32      &>out/cfm/chopsticks/normal.out
python -m plb.algorithms.cfm.train_cfm -d chopsticks -b 32 -c 1 &>out/cfm/chopsticks/simple_mlp.out
python -m plb.algorithms.cfm.train_cfm -d chopsticks -b 32 -c 2 &>out/cfm/chopsticks/linear.out
python -m plb.algorithms.cfm.train_cfm -d chopsticks -b 32 -l OriginInfoNCE &>out/cfm/chopsticks/origin_loss.out

python -m plb.algorithms.cfm.train_cfm -d rope -b 32      &>out/cfm/rope/normal.out
python -m plb.algorithms.cfm.train_cfm -d rope -b 32 -c 1 &>out/cfm/rope/simple_mlp.out
python -m plb.algorithms.cfm.train_cfm -d rope -b 32 -c 2 &>out/cfm/rope/linear.out
python -m plb.algorithms.cfm.train_cfm -d rope -b 32 -l OriginInfoNCE &>out/cfm/rope/origin_loss.out

python -m plb.algorithms.cfm.train_cfm -d torus -b 32      &>out/cfm/torus/normal.out
python -m plb.algorithms.cfm.train_cfm -d torus -b 32 -c 1 &>out/cfm/torus/simple_mlp.out
python -m plb.algorithms.cfm.train_cfm -d torus -b 32 -c 2 &>out/cfm/torus/linear.out
python -m plb.algorithms.cfm.train_cfm -d torus -b 32 -l OriginInfoNCE &>out/cfm/torus/origin_loss.out

python -m plb.algorithms.cfm.train_cfm -d writer -b 32      &>out/cfm/writer/normal.out
python -m plb.algorithms.cfm.train_cfm -d writer -b 32 -c 1 &>out/cfm/writer/simple_mlp.out
python -m plb.algorithms.cfm.train_cfm -d writer -b 32 -c 2 &>out/cfm/writer/linear.out
python -m plb.algorithms.cfm.train_cfm -d writer -b 32 -l OriginInfoNCE &>out/cfm/writer/origin_loss.out

