from PIL import Image
from ovis.serve.runner import RunnerArguments, OvisRunner
image = Image.open('./normal.png')
text = 'Is there any anomaly in the image? Answer in a single word of yes or no.'
runner_args = RunnerArguments(model_path='AIDC-AI/Ovis2-8B')
runner = OvisRunner(runner_args)
generation = runner.run([image, text])
print(generation)