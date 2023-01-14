import { Avatar, FormControl, TextField, Button } from "@mui/material";
import React, { useState } from "react";
import crow from "../../images/crow.png";
import "./toolbar.scss";
export function Nav({
  server,
  setServer,
  darkmode,
  setDarkmode,
  messages,
  setMessages,
  state,
  setState,
}: any) {
  const [isConnected, setIsConnected] = useState(true);

  const exportData = () => {
    const jsonString = `data:text/json;chatset=utf-8,${encodeURIComponent(
      JSON.stringify({ messages: messages, state: state })
    )}`;
    const link = document.createElement("a");
    link.href = jsonString;
    link.download = "data.json";

    link.click();
  };

  const loadData = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const data = JSON.parse(e.target?.result as string);
        setMessages(data.messages);
        setState(data.state);
      };
      reader.readAsText(file);
    }
  };

  const onServerChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setServer(event.target.value);
    // ping the server
    fetch(event.target.value + "/ping")
      .then((_res) => {
        setIsConnected(true);
      })
      .catch((_err) => {
        setIsConnected(false);
      });
  };

  const onThemeClick = () => {
    setDarkmode(!darkmode);
  };
  return (
    <div className="nav">
      <Avatar
        src={crow}
        sx={{
          filter: `invert(${darkmode ? 1 : 0})`,
        }}
        className="main-logo"
        onClick={onThemeClick}
      />
      <h1>Chat-RWKV</h1>
      <FormControl className="server-settings">
        <TextField
          id="serverSet"
          label="Server"
          variant="outlined"
          value={server}
          onChange={onServerChange}
          color="primary"
          sx={{
            marginBottom: 2,
          }}
          InputProps={{
            endAdornment: (
              // Green circle emoji if connected else red circle
              <div>{isConnected ? "🟢" : "🔴"}</div>
            ),
          }}
        />
      </FormControl>
      <br></br>
      <Button variant="outlined" onClick={exportData}>
        Save
      </Button>
      <input type="file" id="convload" hidden onChange={loadData} />
      <Button
        variant="outlined"
        onClick={() => {
          document.getElementById("convload")?.click();
        }}
      >
        Load
      </Button>
    </div>
  );
}
