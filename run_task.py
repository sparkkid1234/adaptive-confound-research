import luigi
import adaptive_confound.pipeline as acp

dataset = "/data/virgile/confound/adaptive/in/twitter_dataset_y=location_z=gender.pkl"
main_task = acp.MainTask(d=dataset)
luigi.build([main_task])