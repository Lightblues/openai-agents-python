""" 
见 [[openai_agents.md#框架设计]]
"""

import abc
import enum
import asyncio
import dataclasses
from collections.abc import Awaitable, AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Generic, Literal, cast, Union
from typing_extensions import TypeAlias, TypedDict, TypeVar
from pydantic import BaseModel, TypeAdapter

from openai.types.responses import Response, ResponseInputItemParam, ResponseOutputItem, ResponseStreamEvent, ResponseOutputMessage, ResponseFunctionToolCall
from openai.types.responses import ResponseComputerToolCall, ResponseFileSearchToolCall, ResponseFunctionWebSearch
from openai.types.responses.response_input_item_param import ComputerCallOutput, FunctionCallOutput
from openai.types.responses.response_reasoning_item import ResponseReasoningItem

TResponse = Response
TResponseInputItem = ResponseInputItemParam
TResponseOutputItem = ResponseOutputItem
TResponseStreamEvent = ResponseStreamEvent


# RunContext: 运行时数据
T = TypeVar("T")
MaybeAwaitable = Union[Awaitable[T], T]
TContext = TypeVar("TContext", default=Any)

@dataclass
class RunContextWrapper(Generic[TContext]):
    context: TContext
    usage: Usage = field(default_factory=Usage)




""" --------------------------------------------------------------------------------------------------------------------
Tool: 工具定义
    工具定义: name, description, params_json_schema
    工具调用: on_invoke_tool
    输入输出: 均为 str
-------------------------------------------------------------------------------------------------------------------- """
Tool = Union[FunctionTool, FileSearchTool, WebSearchTool, ComputerTool]

@dataclass
class FunctionTool:
    name: str
    description: str
    params_json_schema: dict[str, Any]
    on_invoke_tool: Callable[[RunContextWrapper[Any], str], Awaitable[Any]]
    strict_json_schema: bool = True


from openai.types.responses.file_search_tool_param import Filters, RankingOptions
from openai.types.responses.web_search_tool_param import UserLocation
@dataclass
class ComputerTool:
    computer: Computer | AsyncComputer
    def name(self): ...
@dataclass
class FileSearchTool:
    vector_store_ids: list[str]
    max_num_results: int | None = None
    include_search_results: bool = False
    ranking_options: RankingOptions | None = None
    filters: Filters | None = None
    def name(self): ...
@dataclass
class WebSearchTool:
    user_location: UserLocation | None = None
    search_context_size: Literal["low", "medium", "high"] = "medium"
    def name(self): ...
# 辅助: 对于computer的抽象
Environment = Literal["mac", "windows", "ubuntu", "browser"]
Button = Literal["left", "right", "wheel", "back", "forward"]
class Computer(abc.ABC): ...
class AsyncComputer(abc.ABC): ...


@dataclass(init=False)
class AgentOutputSchema:
    _type_adapter: TypeAdapter[Any]
    _is_wrapped: bool
    _output_schema: dict[str, Any]
    strict_json_schema: bool
    def __init__(self, output_type: type[Any], strict_json_schema: bool = True): ...
    def is_plain_text(self) -> bool: ...
    def json_schema(self) -> dict[str, Any]: ...
    def validate_json(self, json_str: str, partial: bool = False) -> Any: ...
    def output_type_name(self) -> str: ...


""" --------------------------------------------------------------------------------------------------------------------
Handoff: 
    对于模型而言, 可以看作一个特殊的工具.
    提供一个 handoff 方法将agent转换 Handoff
-------------------------------------------------------------------------------------------------------------------- """
THandoffInput = TypeVar("THandoffInput", default=Any)

class Handoff(Generic[TContext]):
    tool_name: str
    tool_description: str
    input_json_schema: dict[str, Any]
    on_invoke_handoff: Callable[[RunContextWrapper[Any], str], Awaitable[Agent[TContext]]]
    agent_name: str
    input_filter: HandoffInputFilter | None = None
    strict_json_schema: bool = True
    def get_transfer_message(self, agent: Agent[Any]) -> str: ...
    @classmethod
    def default_tool_name(cls, agent: Agent[Any]) -> str: ...
    @classmethod
    def default_tool_description(cls, agent: Agent[Any]) -> str: ...

@dataclass(frozen=True)
class HandoffInputData:
    input_history: str | tuple[TResponseInputItem, ...]
    pre_handoff_items: tuple[RunItem, ...]
    new_items: tuple[RunItem, ...]

HandoffInputFilter: TypeAlias = Callable[[HandoffInputData], HandoffInputData]

OnHandoffWithInput = Callable[[RunContextWrapper[Any], THandoffInput], Any]
OnHandoffWithoutInput = Callable[[RunContextWrapper[Any]], Any]
def handoff(agent: Agent[TContext], tool_name_override: str | None = None, tool_description_override: str | None = None, on_handoff: OnHandoffWithInput[THandoffInput] | OnHandoffWithoutInput | None = None, input_type: type[THandoffInput] | None = None, input_filter: Callable[[HandoffInputData], HandoffInputData] | None = None) -> Handoff[TContext]: ...


""" --------------------------------------------------------------------------------------------------------------------
Model: agents 场景下抽象 responses API

输入: SP, 增量的input
输出:
    非流式: ModelResponse, 同 responses API (ResponseOutputItem)
    流式: 同 responses API (ResponseStreamEvent)
-------------------------------------------------------------------------------------------------------------------- """
class Model(abc.ABC):
    @abc.abstractmethod
    async def get_response(self, system_instructions: str | None, input: str | list[TResponseInputItem], model_settings: ModelSettings, tools: list[Tool], output_schema: AgentOutputSchema | None, handoffs: list[Handoff], tracing: ModelTracing) -> ModelResponse: ...
    @abc.abstractmethod
    def stream_response(self, system_instructions: str | None, input: str | list[TResponseInputItem], model_settings: ModelSettings, tools: list[Tool], output_schema: AgentOutputSchema | None, handoffs: list[Handoff], tracing: ModelTracing) -> AsyncIterator[TResponseStreamEvent]: ...
# 对于两类API的实现
class OpenAIChatCompletionsModel(Model): ...
class OpenAIResponsesModel(Model): ...


@dataclass
class ModelResponse:
    output: list[TResponseOutputItem]
    usage: Usage
    referenceable_id: str | None
    def to_input_items(self) -> list[TResponseInputItem]: ...

# 提供模型
class ModelProvider(abc.ABC):
    @abc.abstractmethod
    def get_model(self, model_name: str | None) -> Model: ...
class OpenAIProvider(ModelProvider): ...


@dataclass
class ModelSettings:
    temperature: float | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    tool_choice: Literal["auto", "required", "none"] | str | None = None
    parallel_tool_calls: bool | None = False
    truncation: Literal["auto", "disabled"] | None = None
    max_tokens: int | None = None
    def resolve(self, override: ModelSettings | None) -> ModelSettings: ...


class ModelTracing(enum.Enum):
    DISABLED = 0
    ENABLED = 1
    ENABLED_WITHOUT_DATA = 2
    def is_disabled(self) -> bool: ...
    def include_data(self) -> bool: ...

@dataclass
class Usage:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    def add(self, other: "Usage") -> None: ...


""" --------------------------------------------------------------------------------------------------------------------
Agent: 对于一个 agent 的封装
-------------------------------------------------------------------------------------------------------------------- """
@dataclass
class Agent(Generic[TContext]):
    name: str
    instructions: (str | Callable[[RunContextWrapper[TContext], Agent[TContext]], MaybeAwaitable[str]]) | None = None
    handoff_description: str | None = None
    handoffs: list[Agent[Any] | Handoff[TContext]] = field(default_factory=list)
    model: str | Model | None = None
    model_settings: ModelSettings = field(default_factory=ModelSettings)
    tools: list[Tool] = field(default_factory=list)
    input_guardrails: list[InputGuardrail[TContext]] = field(default_factory=list)
    output_guardrails: list[OutputGuardrail[TContext]] = field(default_factory=list)
    output_type: type[Any] | None = None
    hooks: AgentHooks[TContext] | None = None
    tool_use_behavior: (Literal["run_llm_again", "stop_on_first_tool"] | StopAtTools | ToolsToFinalOutputFunction) = "run_llm_again"

    def clone(self, **kwargs: Any) -> Agent[TContext]:
        return dataclasses.replace(self, **kwargs)

class StopAtTools(TypedDict):
    stop_at_tool_names: list[str]

@dataclass
class ToolsToFinalOutputResult:
    is_final_output: bool
    final_output: Any | None = None

ToolsToFinalOutputFunction: TypeAlias = Callable[[RunContextWrapper[TContext], list[FunctionToolResult]], MaybeAwaitable[ToolsToFinalOutputResult]]

# --------------------------------------------------------------------------------
# AgentHooks: 建模agent的生命周期
# --------------------------------------------------------------------------------
# 提供了 agent/tool 两种粒度的 hook
class AgentHooks(Generic[TContext]):
    async def on_start(self, context: RunContextWrapper[TContext], agent: Agent[TContext]) -> None: ...
    async def on_end(self, context: RunContextWrapper[TContext], agent: Agent[TContext], output: Any) -> None: ...
    async def on_handoff(self, context: RunContextWrapper[TContext], agent: Agent[TContext], source: Agent[TContext]) -> None: ...
    async def on_tool_start(self, context: RunContextWrapper[TContext], agent: Agent[TContext], tool: Tool) -> None: ...
    async def on_tool_end(self, context: RunContextWrapper[TContext], agent: Agent[TContext], tool: Tool, result: str) -> None: ...

# 相较于 AgentHooks, 仅仅 on_handoff 参数多了 from_agent 和 to_agent
class RunHooks(Generic[TContext]):
    async def on_agent_start(self, context: RunContextWrapper[TContext], agent: Agent[TContext]) -> None: ...
    async def on_agent_end(self, context: RunContextWrapper[TContext], agent: Agent[TContext], output: Any) -> None: ...
    async def on_handoff(self, context: RunContextWrapper[TContext], agent: Agent[TContext], from_agent: Agent[TContext], to_agent: Agent[TContext]) -> None: ...
    async def on_tool_start(self, context: RunContextWrapper[TContext], agent: Agent[TContext], tool: Tool) -> None: ...
    async def on_tool_end(self, context: RunContextWrapper[TContext], agent: Agent[TContext], tool: Tool, result: str) -> None: ...


# --------------------------------------------------------------------------------
# Guardrail: 
# --------------------------------------------------------------------------------
@dataclass
class InputGuardrail(Generic[TContext]):
    guardrail_function: Callable[[RunContextWrapper[TContext], Agent[Any], str | list[TResponseInputItem]], MaybeAwaitable[GuardrailFunctionOutput]]
    name: str | None = None
    def get_name(self) -> str: ...
    async def run(self, agent: Agent[Any], input: str | list[TResponseInputItem], context: RunContextWrapper[TContext]) -> InputGuardrailResult: ...

@dataclass
class OutputGuardrail(Generic[TContext]):
    guardrail_function: Callable[[RunContextWrapper[TContext], Agent[Any], Any], MaybeAwaitable[GuardrailFunctionOutput]]
    name: str | None = None
    def get_name(self) -> str: ...
    async def run(self, context: RunContextWrapper[TContext], agent: Agent[Any], agent_output: Any) -> OutputGuardrailResult: ...

@dataclass
class GuardrailFunctionOutput:
    output_info: Any
    tripwire_triggered: bool

@dataclass
class InputGuardrailResult:
    guardrail: InputGuardrail[Any]
    output: GuardrailFunctionOutput

@dataclass
class OutputGuardrailResult:
    guardrail: OutputGuardrail[Any]
    agent_output: Any
    agent: Agent[Any]
    output: GuardrailFunctionOutput




""" --------------------------------------------------------------------------------------------------------------------
Runner: 整体运行入口
    非流式: (调用关系) run(...) -> RunResult
        _run_single_turn(...) -> SingleStepResult -- 单步
        _get_new_response(...) -> ModelResponse -- 调用模型 Model.get_response()
    流式: (调用关系) run_streamed(...) -> RunResultStreaming
        _run_streamed_impl(...) -> None -- 调用 RunImpl
        _run_single_turn_streamed(...) -> SingleStepResult
    共用:
        _get_single_step_result_from_response: 调用 RunImpl.process_model_response() 和 RunImpl.execute_tools_and_side_effects(), 得到 SingleStepResult
-------------------------------------------------------------------------------------------------------------------- """
class Runner:
    @classmethod
    async def run(cls, starting_agent: Agent[TContext], input: str | list[TResponseInputItem], *, context: TContext | None = None, max_turns: int = DEFAULT_MAX_TURNS, hooks: RunHooks[TContext] | None = None, run_config: RunConfig | None = None) -> RunResult: ...
    @classmethod
    def run_sync(cls, starting_agent: Agent[TContext], input: str | list[TResponseInputItem], *, context: TContext | None = None, max_turns: int = DEFAULT_MAX_TURNS, hooks: RunHooks[TContext] | None = None, run_config: RunConfig | None = None) -> RunResult: ...
    @classmethod
    def run_streamed(cls, starting_agent: Agent[TContext], input: str | list[TResponseInputItem], *, context: TContext | None = None, max_turns: int = DEFAULT_MAX_TURNS, hooks: RunHooks[TContext] | None = None, run_config: RunConfig | None = None) -> RunResultStreaming: ...

    # 非流式
    @classmethod 
    async def _run_single_turn(cls, *, agent: Agent[Any], original_input: str | list[TResponseInputItem], generated_items: list[RunItem], hooks: RunHooks[TContext], context_wraper: RunContextWrapper[TContext], run_config: RunConfig, should_run_agent_start_hooks: bool) -> SingleStepResult: ...
    @classmethod 
    async def _run_input_guardrails(cls, agent: Agent[Any], guardrails: list[InputGuardrail[TContext]], input: str | list[TResponseInputItem], context: RunContextWrapper[TContext]) -> list[InputGuardrailResult]: ...
    @classmethod # 调用 Model.get_response()
    async def _get_new_response(cls, agent: Agent[TContext], system_prompt: str | None, input: list[TResponseInputItem], output_schema: AgentOutputSchema | None, handoffs: list[Handoff], context_wrapper: RunContextWrapper[TContext], run_config: RunConfig) -> ModelResponse: ...

    # 流式
    @classmethod # 调用 RunImpl.stream_step_result_to_queue(single_step_result, streamed_result._event_queue)
    async def _run_single_turn_streamed(cls, streamed_result: RunResultStreaming, agent: Agent[TContext], hooks: RunHooks[TContext], context_wrapper: RunContextWrapper[TContext], run_config: RunConfig, should_run_agent_start_hooks: bool) -> SingleStepResult: ...
    @classmethod # 并发调用 RunImpl.run_single_input_guardrail(agent, guardrail, input, context)
    async def _run_input_guardrails_with_queue(cls, agent: Agent[Any], guardrails: list[InputGuardrail[TContext]], input: str | list[TResponseInputItem], context: RunContextWrapper[TContext], streamed_result: RunResultStreaming, parent_span: Span[Any]): ...
    @classmethod
    async def _run_streamed_impl(cls, starting_input: str | list[TResponseInputItem], streamed_result: RunResultStreaming, starting_agent: Agent[TContext], max_turns: int, hooks: RunHooks[TContext], context_wrapper: RunContextWrapper[TContext], run_config: RunConfig): ...

    # 下面是流式/非流式共用的
    @classmethod # 并发调用 RunImpl.run_single_output_guardrail(guardrail, agent, agent_output, context)
    async def _run_output_guardrails(cls, guardrails: list[OutputGuardrail[TContext]], agent: Agent[TContext], agent_output: Any, context: RunContextWrapper[TContext]) -> list[OutputGuardrailResult]: ...
    @classmethod # 调用 RunImpl.process_model_response() 和 RunImpl.execute_tools_and_side_effects(), 得到 SingleStepResult
    async def _get_single_step_result_from_response(cls, *, agent: Agent[TContext], original_input: str | list[TResponseInputItem], pre_step_items: list[RunItem], new_response: ModelResponse, output_schema: AgentOutputSchema | None, handoffs: list[Handoff], hooks: RunHooks[TContext], context_wrapper: RunContextWrapper[TContext], run_config: RunConfig) -> SingleStepResult: ...
    @classmethod
    def _get_output_schema(cls, agent: Agent[Any]) -> AgentOutputSchema | None: ...
    @classmethod
    def _get_handoffs(cls, agent: Agent[Any]) -> list[Handoff]: ...
    @classmethod
    def _get_model(cls, agent: Agent[Any], run_config: RunConfig) -> Model: ...


# --------------------------------------------------------------------------------
# RunResult
# --------------------------------------------------------------------------------
@dataclass
class RunResultBase(abc.ABC):
    input: str | list[TResponseInputItem]
    new_items: list[RunItem]
    raw_responses: list[ModelResponse]
    final_output: Any
    input_guardrail_results: list[InputGuardrailResult]
    output_guardrail_results: list[OutputGuardrailResult]
    @property
    def last_agent(self) -> Agent[Any]: ...
    def final_output_as(self, cls: type[T], raise_if_incorrect_type: bool = False) -> T: ...
    def to_input_list(self) -> list[TResponseInputItem]: ...

@dataclass
class RunResult(RunResultBase):
    @property
    def last_agent(self) -> Agent[Any]: ...
@dataclass
class RunResultStreaming(RunResultBase):
    current_agent: Agent[Any]
    current_turn: int
    max_turns: int
    final_output: Any
    is_complete: bool = False
    @property
    def last_agent(self) -> Agent[Any]: ...
    async def stream_events(self) -> AsyncIterator[StreamEvent]: ...


# --------------------------------------------------------------------------------
# StepResult: 建模run过程中一步. 
#   next_step: 包括 handoff, final output, run again
# --------------------------------------------------------------------------------
@dataclass
class SingleStepResult:
    original_input: str | list[TResponseInputItem]
    model_response: ModelResponse
    pre_step_items: list[RunItem]
    new_step_items: list[RunItem]
    next_step: NextStepHandoff | NextStepFinalOutput | NextStepRunAgain
    @property
    def generated_items(self) -> list[RunItem]: ...

@dataclass
class NextStepHandoff:
    new_agent: Agent[Any]
@dataclass
class NextStepFinalOutput:
    output: Any
@dataclass
class NextStepRunAgain:
    pass


# --------------------------------------------------------------------------------
# RunConfig
# --------------------------------------------------------------------------------
DEFAULT_MAX_TURNS = 10


@dataclass
class RunConfig:
    model: str | Model | None = None
    model_provider: ModelProvider = field(default_factory=OpenAIProvider)
    model_settings: ModelSettings | None = None
    handoff_input_filter: HandoffInputFilter | None = None
    input_guardrails: list[InputGuardrail[Any]] | None = None
    output_guardrails: list[OutputGuardrail[Any]] | None = None
    tracing_disabled: bool = False
    trace_include_sensitive_data: bool = True
    workflow_name: str = "Agent workflow"
    trace_id: str | None = None
    group_id: str | None = None
    trace_metadata: dict[str, Any] | None = None

""" --------------------------------------------------------------------------------------------------------------------
RunImpl: 运行实现
    顶层接口:
        execute_tools_and_side_effects(...) -> SingleStepResult 执行工具
            execute_function_tool_calls | execute_computer_actions | execute_handoffs
        run_single_input_guardrail(...) -> InputGuardrailResult
        run_single_output_guardrail(...) -> OutputGuardrailResult
    辅助函数:
        process_model_response(agent, response, output_schema, handoffs) -> ProcessedResponse. 转为 RunImpl 执行过程的中间数据 (用于 execute_tools_and_side_effects)
-------------------------------------------------------------------------------------------------------------------- """
class RunImpl:
    @classmethod # 会调用 cls.execute_function_tool_calls | cls.execute_computer_actions | cls.execute_handoffs
    async def execute_tools_and_side_effects(cls, *, agent: Agent[TContext], original_input: str | list[TResponseInputItem], pre_step_items: list[RunItem], new_response: ModelResponse, processed_response: ProcessedResponse, output_schema: AgentOutputSchema | None, hooks: RunHooks[TContext], context_wrapper: RunContextWrapper[TContext], run_config: RunConfig) -> SingleStepResult: ...
    @classmethod
    def process_model_response(cls, *, agent: Agent[Any], response: ModelResponse, output_schema: AgentOutputSchema | None, handoffs: list[Handoff]) -> ProcessedResponse: ...
    
    @classmethod
    async def execute_function_tool_calls(cls, *, agent: Agent[TContext], tool_runs: list[ToolRunFunction], hooks: RunHooks[TContext], context_wrapper: RunContextWrapper[TContext], config: RunConfig) -> list[FunctionToolResult]: ...
    @classmethod
    async def execute_computer_actions(cls, *, agent: Agent[TContext], actions: list[ToolRunComputerAction], hooks: RunHooks[TContext], context_wrapper: RunContextWrapper[TContext], config: RunConfig) -> list[RunItem]: ...
    @classmethod
    async def execute_handoffs(cls, *, agent: Agent[TContext], original_input: str | list[TResponseInputItem], pre_step_items: list[RunItem], new_step_items: list[RunItem], new_response: ModelResponse, run_handoffs: list[ToolRunHandoff], hooks: RunHooks[TContext], context_wrapper: RunContextWrapper[TContext], run_config: RunConfig) -> SingleStepResult: ...
    @classmethod # -> cls.run_final_output_hooks
    async def execute_final_output(cls, *, agent: Agent[TContext], original_input: str | list[TResponseInputItem], new_response: ModelResponse, pre_step_items: list[RunItem], hooks: RunHooks[TContext], context_wrapper: RunContextWrapper[TContext], run_config: RunConfig) -> SingleStepResult: ...
    @classmethod
    async def run_final_output_hooks(cls, agent: Agent[TContext], hooks: RunHooks[TContext], context_wrapper: RunContextWrapper[TContext], final_output: Any): ...
    
    @classmethod
    async def run_single_input_guardrail(cls, agent: Agent[Any], guardrail: InputGuardrail[TContext], input: str | list[TResponseInputItem], context: RunContextWrapper[TContext]) -> InputGuardrailResult: ...
    @classmethod
    async def run_single_output_guardrail(cls, guardrail: OutputGuardrail[TContext], agent: Agent[Any], agent_output: Any, context: RunContextWrapper[TContext]) -> OutputGuardrailResult: ...

    @classmethod # 将流式的一个 step result 中的信息放到 queue 中
    def stream_step_result_to_queue(cls, step_result: SingleStepResult, queue: asyncio.Queue[StreamEvent | QueueCompleteSentinel]): ...
    @classmethod
    async def _check_for_final_output_from_tools(cls, *, agent: Agent[TContext], tool_results: list[FunctionToolResult], context_wrapper: RunContextWrapper[TContext], config: RunConfig) -> ToolsToFinalOutputResult: ...


@dataclass
class ProcessedResponse:
    new_items: list[RunItem]
    handoffs: list[ToolRunHandoff]
    functions: list[ToolRunFunction]
    computer_actions: list[ToolRunComputerAction]
@dataclass
class ToolRunHandoff:
    handoff: Handoff
    tool_call: ResponseFunctionToolCall
@dataclass
class ToolRunFunction:
    tool_call: ResponseFunctionToolCall
    function_tool: FunctionTool
@dataclass
class ToolRunComputerAction:
    tool_call: ResponseComputerToolCall
    computer_tool: ComputerTool

class QueueCompleteSentinel: # 队列的空元素?
    pass

# RunImpl.execute_function_tool_calls(...) -> FunctionToolResult
@dataclass
class FunctionToolResult:
    tool: FunctionTool
    output: Any
    run_item: RunItem

# --------------------------------------------------------------------------------
# StreamEvent: 对于 Runner/agent 流式行为的建模
# --------------------------------------------------------------------------------
StreamEvent: TypeAlias = Union[RawResponsesStreamEvent, RunItemStreamEvent, AgentUpdatedStreamEvent]
@dataclass
class RawResponsesStreamEvent:
    data: TResponseStreamEvent
    type: Literal["raw_response_event"] = "raw_response_event"
@dataclass
class RunItemStreamEvent:
    name: Literal["message_output_created", "handoff_requested", "handoff_occured", "tool_called", "tool_output", "reasoning_item_created"]
    item: RunItem
    type: Literal["run_item_stream_event"] = "run_item_stream_event"
@dataclass
class AgentUpdatedStreamEvent:
    new_agent: Agent[Any]
    type: Literal["agent_updated_stream_event"] = "agent_updated_stream_event"

T = TypeVar("T", bound=Union[TResponseOutputItem, TResponseInputItem])
@dataclass
class RunItemBase(Generic[T], abc.ABC):
    agent: Agent[Any]
    raw_item: T
    def to_input_item(self) -> TResponseInputItem: ...

@dataclass
class MessageOutputItem(RunItemBase[ResponseOutputMessage]):
    raw_item: ResponseOutputMessage
    type: Literal["message_output_item"] = "message_output_item"
@dataclass
class HandoffCallItem(RunItemBase[ResponseFunctionToolCall]):
    raw_item: ResponseFunctionToolCall
    type: Literal["handoff_call_item"] = "handoff_call_item"
@dataclass
class HandoffOutputItem(RunItemBase[TResponseInputItem]):
    raw_item: TResponseInputItem
    source_agent: Agent[Any]
    target_agent: Agent[Any]
    type: Literal["handoff_output_item"] = "handoff_output_item"

ToolCallItemTypes: TypeAlias = Union[ResponseFunctionToolCall, ResponseComputerToolCall, ResponseFileSearchToolCall, ResponseFunctionWebSearch]
@dataclass
class ToolCallItem(RunItemBase[ToolCallItemTypes]):
    raw_item: ToolCallItemTypes
    type: Literal["tool_call_item"] = "tool_call_item"
@dataclass
class ToolCallOutputItem(RunItemBase[Union[FunctionCallOutput, ComputerCallOutput]]):
    raw_item: FunctionCallOutput | ComputerCallOutput
    output: Any
    type: Literal["tool_call_output_item"] = "tool_call_output_item"
@dataclass
class ReasoningItem(RunItemBase[ResponseReasoningItem]):
    raw_item: ResponseReasoningItem
    type: Literal["reasoning_item"] = "reasoning_item"

RunItem: TypeAlias = Union[MessageOutputItem, HandoffCallItem, HandoffOutputItem, ToolCallItem, ToolCallOutputItem, ReasoningItem]



""" --------------------------------------------------------------------------------------------------------------------
Tracing
-------------------------------------------------------------------------------------------------------------------- """
TSpanData = TypeVar("TSpanData", bound=SpanData)

class Span(abc.ABC, Generic[TSpanData]):
    def trace_id(self) -> str: ...
    def span_id(self) -> str: ...
    def span_data(self) -> TSpanData: ...
    def start(self, mark_as_current: bool = False): ...
    def finish(self, reset_current: bool = False) -> None: ...
    def __enter__(self) -> Span[TSpanData]: ...
    def __exit__(self, exc_type, exc_val, exc_tb): ...
    def parent_id(self) -> str | None: ...
    def set_error(self, error: SpanError) -> None: ...
    def error(self) -> SpanError | None: ...
    def export(self) -> dict[str, Any] | None: ...
    def started_at(self) -> str | None: ...
    def ended_at(self) -> str | None: ...

class SpanError(TypedDict):
    message: str
    data: dict[str, Any] | None

class SpanData(abc.ABC):
    def export(self) -> dict[str, Any]: ...
    def type(self) -> str: ...
