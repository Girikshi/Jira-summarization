from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


class JiraParser(BaseModel):
    jira_id: str = Field(description="Jira ID")
    title: str = Field(description="Title")
    assignee: str = Field(description="Assignee")
    status: str = Field(description="Status")
    summary: str = Field(description="summary")

    def to_dict(self):
        return {
            "jira_id": self.jira_id,
            "title": self.title,
            "assignee": self.assignee,
            "status": self.status,
            "summary": self.summary
        }


jira_parser: PydanticOutputParser = PydanticOutputParser(
    pydantic_object=JiraParser
)
