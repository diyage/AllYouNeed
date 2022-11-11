import typing


class FirstLevelCollection:
    def __init__(self):
        self._store_info = {}

    def at(
            self,
            info_key: str
    ) -> list:
        return self._store_info[info_key]

    def put_info(
            self,
            info_key: str,
            info: typing.Union[int, float]
    ):
        if self._store_info.get(info_key) is None:
            self._store_info[info_key] = []
        self._store_info[info_key].append(info)

    def get_info(
            self,
            info_key: str,
    ) -> typing.Union[int, float]:
        if self._store_info.get(info_key) is None:
            raise RuntimeError
        else:
            val = sum(self._store_info[info_key]) / len(self._store_info[info_key])
            return val

    def get_keys(self):
        return list(self._store_info.keys())


class SecondLevelCollection:
    def __init__(self):
        self._store_info = {}

    def at(
            self,
            batch_id: str,
    ) -> FirstLevelCollection:
        return self._store_info[batch_id]

    def put_info(
            self,
            batch_id: str,
            info_key: str,
            info: typing.Union[int, float]
    ):
        if self._store_info.get(batch_id) is None:
            self._store_info[batch_id] = FirstLevelCollection()

        self._store_info[batch_id].put_info(info_key, info)

    def get_info(
            self,
            batch_id: str,
            info_key: str,

    ) -> typing.Union[int, float]:

        if self._store_info.get(batch_id) is None:
            raise RuntimeError
        else:
            val = self._store_info[batch_id].get_info(info_key)
            return val

    def get_keys(self):
        return list(self._store_info.keys())


class ThirdLevelCollection:
    def __init__(self):
        self._store_info = {}

    def at(
            self,
            epoch_id: str,
    ) -> SecondLevelCollection:
        return self._store_info[epoch_id]

    def put_info(
            self,
            epoch_id: str,
            batch_id: str,
            info_key: str,
            info: typing.Union[int, float]
    ):
        if self._store_info.get(epoch_id) is None:
            self._store_info[epoch_id] = SecondLevelCollection()
        self._store_info[epoch_id].put_info(batch_id, info_key, info)

    def get_info(
            self,
            epoch_id: str,
            info_key: str,
    ) -> typing.Union[int, float]:

        if self._store_info.get(epoch_id) is None:
            raise RuntimeError
        else:
            second_collect: SecondLevelCollection = self._store_info[epoch_id]
            info_for_each_batch_vec = []

            for batch_id in second_collect.get_keys():
                info = second_collect.get_info(batch_id, info_key)
                info_for_each_batch_vec.append(info)

            return sum(info_for_each_batch_vec)/ len(info_for_each_batch_vec)

    def print_info(
            self,
            epoch_id: str,
    ):
        if self._store_info.get(epoch_id) is None:
            raise RuntimeError
        else:
            second_collect: SecondLevelCollection = self._store_info[epoch_id]
            info_collect_dict = {}

            for batch_id in second_collect.get_keys():
                first_collect: FirstLevelCollection = second_collect.at(batch_id)

                for info_key in first_collect.get_keys():
                    info = first_collect.get_info(info_key)

                    if info_collect_dict.get(info_key) is None:
                        info_collect_dict[info_key] = []

                    info_collect_dict[info_key].append(info)

            print('\n Info epoch: {} --> '.format(epoch_id))
            for info_key, info_vec in info_collect_dict.items():
                if info_key.lower().find('acc') == -1:
                    print('\t {}: {:.5f}'.format(info_key, sum(info_vec) / len(info_vec)))
                else:
                    print('\t {}: {:.2%}'.format(info_key, sum(info_vec) / len(info_vec)))
