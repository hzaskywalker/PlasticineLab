mkdir -p out
mkdir -p out/cfm
mkdir -p out/cfm/chopsticks
mkdir -p out/cfm/rope
mkdir -p out/cfm/torus
mkdir -p out/cfm/writer
mkdir -p out/inverse
mkdir -p out/inverse/chopsticks
mkdir -p out/inverse/rope
mkdir -p out/inverse/torus
mkdir -p out/inverse/writer
mkdir -p out/forward
mkdir -p out/forward/chopsticks
mkdir -p out/forward/rope
mkdir -p out/forward/torus
mkdir -p out/forward/writer
mkdir -p out/e2c
mkdir -p out/e2c/chopsticks
mkdir -p out/e2c/rope
mkdir -p out/e2c/torus
mkdir -p out/e2c/writer

mkdir -p pretrain_model/inverse
mkdir -p pretrain_model/inverse/torus
mkdir -p pretrain_model/inverse/writer
mkdir -p pretrain_model/forward
mkdir -p pretrain_model/forward/torus
mkdir -p pretrain_model/forward/writer
mkdir -p pretrain_model/e2c
mkdir -p pretrain_model/e2c/torus
mkdir -p pretrain_model/e2c/writer

# python -m plb.algorithms.cfm.train_cfm -d chopsticks -b 32      &>out/cfm/chopsticks/normal.out
# python -m plb.algorithms.cfm.train_cfm -d chopsticks -b 32 -c 1 &>out/cfm/chopsticks/simple_mlp.out
# python -m plb.algorithms.cfm.train_cfm -d chopsticks -b 32 -c 2 &>out/cfm/chopsticks/linear.out
# python -m plb.algorithms.cfm.train_cfm -d chopsticks -b 32 -l OriginInfoNCE &>out/cfm/chopsticks/origin_loss.out

# python -m plb.algorithms.cfm.train_cfm -d rope -b 32      &>out/cfm/rope/normal.out
# python -m plb.algorithms.cfm.train_cfm -d rope -b 32 -c 1 &>out/cfm/rope/simple_mlp.out
# python -m plb.algorithms.cfm.train_cfm -d rope -b 32 -c 2 &>out/cfm/rope/linear.out
# python -m plb.algorithms.cfm.train_cfm -d rope -b 32 -l OriginInfoNCE &>out/cfm/rope/origin_loss.out

# python -m plb.algorithms.cfm.train_cfm -d torus -b 32      &>out/cfm/torus/normal.out
# python -m plb.algorithms.cfm.train_cfm -d torus -b 32 -c 1 &>out/cfm/torus/simple_mlp.out
# python -m plb.algorithms.cfm.train_cfm -d torus -b 32 -c 2 &>out/cfm/torus/linear.out
# python -m plb.algorithms.cfm.train_cfm -d torus -b 32 -l OriginInfoNCE &>out/cfm/torus/origin_loss.out

# python -m plb.algorithms.cfm.train_cfm -d writer -b 32      &>out/cfm/writer/normal.out
# python -m plb.algorithms.cfm.train_cfm -d writer -b 32 -c 1 &>out/cfm/writer/simple_mlp.out
# python -m plb.algorithms.cfm.train_cfm -d writer -b 32 -c 2 &>out/cfm/writer/linear.out
# python -m plb.algorithms.cfm.train_cfm -d writer -b 32 -l OriginInfoNCE &>out/cfm/writer/origin_loss.out

# python -m plb.algorithms.cfm.train_cfm -d chopsticks -b 32 -l Forward &>out/forward/chopsticks/encoder.out
# python -m plb.algorithms.cfm.train_cfm -d chopsticks -b 32 -l Inverse &>out/inverse/chopsticks/encoder.out
# python -m plb.algorithms.cfm.train_cfm -d chopsticks -b 32 -l E2C     &>out/e2c/chopsticks/encoder.out

# python -m plb.algorithms.cfm.train_cfm -d rope -b 32 -l Forward &>out/forward/rope/encoder.out
# python -m plb.algorithms.cfm.train_cfm -d rope -b 32 -l Inverse &>out/inverse/rope/encoder.out
# python -m plb.algorithms.cfm.train_cfm -d rope -b 32 -l E2C     &>out/e2c/rope/encoder.out

python -m plb.algorithms.cfm.train_cfm -d torus -b 32 -l Forward &>out/forward/torus/encoder.out
python -m plb.algorithms.cfm.train_cfm -d torus -b 32 -l Inverse &>out/inverse/torus/encoder.out
python -m plb.algorithms.cfm.train_cfm -d torus -b 32 -l E2C     &>out/e2c/torus/encoder.out

python -m plb.algorithms.cfm.train_cfm -d writer -b 32 -l Forward &>out/forward/writer/encoder.out
python -m plb.algorithms.cfm.train_cfm -d writer -b 32 -l Inverse &>out/inverse/writer/encoder.out
python -m plb.algorithms.cfm.train_cfm -d writer -b 32 -l E2C     &>out/e2c/writer/encoder.out
