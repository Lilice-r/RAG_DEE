import json


class FileUtil(object):
    """
    文件工具类
    """

    @classmethod
    def save_raw_data(cls, raw_list, data_path):
        """
        存储原始数据到文件中
        :param raw_list:
        :param data_path:
        :return:
        """
        with open(data_path, "w", encoding="utf-8") as raw_file:
            for item in raw_list:
                raw_file.write(item + "\n")

    @classmethod
    def save_json_data(cls, data_list, data_path):
        """
        存储原始数据到文件中
        :param data_list:
        :param data_path:
        :return:
        """
        with open(data_path, "w", encoding="utf-8") as raw_file:
            for data in data_list:
                raw_file.write(json.dumps(data) + '\n')


    @classmethod
    def read_json_data(cls, data_path):
        """
        从文件中读取原始数据
        :param data_path:
        :return:
        """
        raw_list = []
        with open(data_path, "r", encoding="utf-8") as raw_file:
            for item in raw_file:
                raw_list.append(json.loads(item.strip()))

        return raw_list



    @classmethod
    def read_raw_data(cls, data_path):
        """
        从文件中读取原始数据
        :param data_path:
        :return:
        """
        raw_list = []
        with open(data_path, "r", encoding="utf-8") as raw_file:
            for item in raw_file:
                raw_list.append(item.strip())

        return raw_list
