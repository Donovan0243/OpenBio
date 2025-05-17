from typing import List, Dict, Union, Generator, Iterator

class Pipeline:
    def __init__(self):
        # Optionally, you can set the id and name of the pipeline.
        # Best practice is not to specify the id so that it can be automatically inferred
        # from the filename, so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens.
        # It cannot contain spaces, special characters, slashes, or backslashes.
        # self.id = "python_code_pipeline"
        self.name = "Pipeline Demo"

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup: {__name__}")

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown: {__name__}")

    def create_datadict(self):
        # Creates a data dictionary.
        datadict = {
            'deepseek': 'DeepSeek is a powerful search engine for deep learning models.',
            'qwen': 'Qwen is an advanced AI model for natural language processing.'
        }
        return datadict

    def pipe(
        self, user_message: str, model_id: str, messages: List[Dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom pipelines like RAG.
        print(f"pipe: {__name__}")

        print(messages)
        print(user_message)

        if body.get("title", False):
            print("Title Generation")
            return "Python Code Pipeline"
        else:
            print("Look up in data frame")
            datadict = self.create_datadict()
            if user_message.lower() not in datadict.keys():
                return "Cannot find product in database. Available products are: {}".format(
                    list(datadict.keys())
                )
            else:
                return datadict[user_message.lower()]
