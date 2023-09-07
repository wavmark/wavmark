import argparse
import ast


class MyParser():

    def __init__(self, epoch=0, batch_size=0, worker=0, seed=2526,
                 max_hour=100, early_stop=5, lr=1e-3,
                 model_save_folder=None):
        super(MyParser, self).__init__()
        parser = argparse.ArgumentParser()
        parser.add_argument("--seed", default=seed, type=int)
        parser.add_argument("--worker", default=worker, type=int)
        parser.add_argument("--epoch", default=epoch, type=int)
        parser.add_argument("--batch_size", default=batch_size, type=int)
        parser.add_argument("--max_hour", default=max_hour, type=int)
        parser.add_argument("--early_stop", default=early_stop, type=int)
        parser.add_argument("--lr", default=lr, type=float)
        parser.add_argument("--model_save_folder", default=model_save_folder, type=str)
        self.core_parser = parser

    def use_wb(self, project, name, dryrun=True):
        self.project = project
        self.name = name
        self.dryrun = dryrun
        parser = self.core_parser
        parser.add_argument("--project", default=self.project, type=str)
        parser.add_argument("--name", default=self.name, type=str)
        parser.add_argument("--dryrun", default=self.dryrun, type=ast.literal_eval)

    def custom(self, the_dict):
        parser = self.core_parser
        for key in the_dict:
            value = the_dict[key]
            if type(value) == str or value is None:
                parser.add_argument("--" + key, default=value, type=str)
            elif type(value) == int:
                parser.add_argument("--" + key, default=value, type=int)
            elif type(value) == float:
                parser.add_argument("--" + key, default=value, type=float)
            elif type(value) == bool:
                parser.add_argument("--" + key, default=value, type=ast.literal_eval)
            else:
                raise Exception("unsupported type:" + type(value))

    def parse(self):
        args = parse_it(self.core_parser)
        return args

    def show(self):
        the_dic = vars(self.parse())
        keys = list(the_dic.keys())
        keys.sort()
        for key in keys:
            print(key, ":", the_dic[key])

    def parse_in_jupyter(self):
        args = self.core_parser.parse_args([])
        return args


def parse_it(parser):
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    pass
