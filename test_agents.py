from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

class MockRetriever(BaseRetriever):
    def get_relevant_documents(self, query):
        return [Document(page_content="Mock document about SRH’s Machine Learning course.")]

    async def aget_relevant_documents(self, query):
        return [Document(page_content="Mock document about SRH’s Machine Learning course.")]


from agents.curriculum_agent import CurriculumAgent

agent = CurriculumAgent(retriever=MockRetriever())
print(agent.run("What is taught in Machine Learning course?"))
