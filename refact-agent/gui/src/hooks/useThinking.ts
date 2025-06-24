import { useCallback, useEffect, useMemo } from "react";
// import { useCapsForToolUse } from "./useCapsForToolUse";
import { useAppSelector } from "./useAppSelector";
import {
  selectChatId,
  // selectIsStreaming,
  // selectIsWaiting,
  selectThreadBoostReasoning,
  setBoostReasoning,
} from "../features/Chat";
import { selectIsStreaming, selectIsWaiting } from "../features/ThreadMessages";
import { useAppDispatch } from "./useAppDispatch";
// import { useGetUser } from "./useGetUser";

export function useThinking() {
  const dispatch = useAppDispatch();

  const isStreaming = useAppSelector(selectIsStreaming);
  const isWaiting = useAppSelector(selectIsWaiting);
  const chatId = useAppSelector(selectChatId);

  const isBoostReasoningEnabled = useAppSelector(selectThreadBoostReasoning);

  // const caps = useCapsForToolUse();
  // const { data: userData } = useGetUser();

  const supportsBoostReasoning = false;
  // const supportsBoostReasoning = useMemo(() => {
  //   const models = caps.data?.chat_models;
  //   const item = models?.[caps.currentModel];
  //   return item?.supports_boost_reasoning ?? false;
  // }, [caps.data?.chat_models, caps.currentModel]);

  const shouldBeTeasing = useMemo(
    () => false /*userData?.inference === "FREE"*/,
    [
      /*userData*/
    ],
  );

  const shouldBeDisabled = useMemo(() => {
    return (
      // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
      !supportsBoostReasoning || shouldBeTeasing || isStreaming || isWaiting
    );
  }, [supportsBoostReasoning, isStreaming, isWaiting, shouldBeTeasing]);

  const noteText = useMemo(() => {
    // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
    if (!supportsBoostReasoning)
      return `Note: current model doesn't support thinking`;
    if (isStreaming || isWaiting)
      return `Note: you can't ${
        isBoostReasoningEnabled ? "disable" : "enable"
      } reasoning while stream is in process`;
  }, [supportsBoostReasoning, isStreaming, isWaiting, isBoostReasoningEnabled]);

  const handleReasoningChange = useCallback(
    (event: React.MouseEvent<HTMLButtonElement>, checked: boolean) => {
      event.stopPropagation();
      event.preventDefault();
      dispatch(setBoostReasoning({ chatId, value: checked }));
    },
    [dispatch, chatId],
  );

  useEffect(() => {
    // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
    if (!supportsBoostReasoning) {
      dispatch(setBoostReasoning({ chatId, value: supportsBoostReasoning }));
    }
  }, [dispatch, chatId, supportsBoostReasoning, shouldBeDisabled]);

  return {
    handleReasoningChange,
    shouldBeDisabled,
    shouldBeTeasing,
    noteText,
    areCapsInitialized: true,
  };
}
