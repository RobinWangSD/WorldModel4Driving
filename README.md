# WorldModel4Driving
cd root
git clone https://github.com/RobinWangSD/WorldModel4Driving.git
pip install uv
cd /root/WorldModel4Driving/le-wm/ && uv venv --python=3.10 && source .venv/bin/activate && uv pip install stable-worldmodel[train,env]
cd /root/WorldModel4Driving/le-wm && uv pip install --python .venv/bin/python "datasets>=2.0"

cp -r /closed-loop-e2e/wm_data /root/ && cd /root/WorldModel4Driving/le-wm && STABLEWM_HOME=/root/wm_data/dataset WANDB_API_KEY=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 .venv/bin/python train.py data=pusht loss.reg.type=gaussian_kl output_model_name=lewm-kl-exp1

cp -r /closed-loop-e2e/wm_data /root/ && cd /root/WorldModel4Driving/le-wm && STABLEWM_HOME=/root/wm_data/dataset WANDB_API_KEY=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 .venv/bin/python train.py data=pusht output_model_name=lewm-sigreg-exp1