import "./message.scss";
import React from "react";
import { Avatar, Box, Paper } from "@mui/material";

export enum MessageType {
  Normal = "normal",
  Success = "success",
  Error = "error",
  Info = "info",
  Warning = "warning",
}

export enum MessageSide {
  Left = "left",
  Right = "right",
}

export interface MessageProps {
  text?: string;
  icon?: string;
  type?: MessageType;
  side?: MessageSide;
  index?: number;
  darkTheme?: boolean;
  timeSent?: Date;
}

function Message({
  text = "No message",
  type = MessageType.Normal,
  side = MessageSide.Left,
  icon = undefined,
  index = 10,
  darkTheme = true,
  timeSent = undefined,
}: MessageProps) {
  return (
    <Paper
      elevation={Math.max(12 - index, 0)}
      className={["message", type, side].join(" ")}
      sx={{
        position: "relative",
        minWidth: 200,
      }}
    >
      {icon ? (
        <Avatar
          src={icon}
          alt="icon"
          sx={{
            width: 32,
            height: 32,
            position: "absolute",
            top: -16,
            filter: `invert(${darkTheme ? 1 : 0})`,
            ...(side === MessageSide.Left ? { left: -16 } : { right: -16 }),
          }}
        />
      ) : (
        ""
      )}
      {text
        .replace("\nEnd", "")
        .split("\n")
        .map((line, i) => (
          <div key={i}>{line.replace("User:", "").replace("END", "")}</div>
        ))}
      {timeSent ? (
        <Box
          className="time-sent"
          sx={{
            position: "absolute",
            bottom: -16,

            ...(side === MessageSide.Left ? { left: 0 } : { right: 0 }),
          }}
        >
          {timeSent.toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit",
          })}
        </Box>
      ) : (
        ""
      )}
    </Paper>
  );
}

export default Message;
