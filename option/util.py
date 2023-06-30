
import configparser
class Settings:
    def __init__(self, file_path):
        self.parse_config_file(file_path)

    def parse_config_file(self, file_path):
        config = configparser.ConfigParser()
        config.read(file_path)

        for option in config['DEFAULT']:
            value = config.get('DEFAULT', option)

            # 特殊处理 datetime
            if option == 'datetime':
                value = str(value)
            else:
                # 尝试自动转换类型
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        if value.lower() == 'true':
                            value = True
                        elif value.lower() == 'false':
                            value = False
                        else:
                            value = str(value)

            setattr(self, option, value)
