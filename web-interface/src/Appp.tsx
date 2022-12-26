import "./App.scss";
import React, { useState } from "react";
import Message, {
  MessageProps,
  MessageSide,
  MessageType,
} from "./components/message/Message";
import TextField from "@mui/material/TextField";
import { Avatar, Button } from "@mui/material";
import { ThemeProvider, createTheme } from "@mui/material/styles";
import CssBaseline from "@mui/material/CssBaseline";
import crow from "./images/crow.png";
// import png

const darkTheme = createTheme({
  palette: {
    mode: "dark",
  },
});

type RwkvState = [number[], number[], number[], number[]] | undefined;

const mess: MessageProps[] = [
  { text: "Hello", type: MessageType.Success, side: MessageSide.Left },
  { text: "World", type: MessageType.Error, side: MessageSide.Right },
  { text: "!", type: MessageType.Info },
];

function App() {
  const [messages, setMessages] = useState(mess);
  const [state, setState] = useState<RwkvState>(undefined);
  const [currentMessage, setCurrentMessage] = useState("");
  // set state to current address
  const [server, setServer] = useState(document.location.href);

  const getMessage = async (message: string, state: RwkvState) => {
    // do a fetch to the server
    const data = await fetch(server, {
      method: "POST",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        message: message,
        state: state,
      }),
    });

    const datajson: { message: string; state: RwkvState } = await data.json();

    setState(datajson.state);
    setMessages([
      ...messages,
      {
        text: message,
        icon: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAOEAAADhCAMAAAAJbSJIAAAAeFBMVEX///8AAACAgIB8fHz19fXp6emNjY37+/umpqbY2NiYmJiwsLB2dnb4+PgwMDCtra2UlJTKysq/v7/R0dGdnZ1eXl4qKipvb2/v7+9ISEhRUVHg4OATExMLCwsiIiJnZ2eIiIhBQUE6OjrDw8NFRUVZWVkbGxu4uLjqLqXEAAAEJElEQVR4nO3di3aiMBAGYOMNEPCCN6pWq1Z9/zfcspazPZZISAZmcP/vCfIfYhKSMHY6AAAAAAAAAAAAAAAAAAAAAK+nPwhCf5jxJ8Ggz90cYvF8sVypn1aX0fx1Us6jnSryvp5xN42CNylMl/Nj7gY68vz904BK7Ratzjgfl+TL7M/czbTmrQ3yZQ4tHXOCq2HAL3Puxtp4PsI8WnA3t7qPSgGVirgbXJXpT/CfJXeTqzlUDtiyiJFFwK8hlbvZ5oZWAZXqcjfclG8ZUKkRd9PNBNYBWzIveiYrNZ19G1Y3dqNMrgWjzdwpoFLyl+EufTSz4w5Qxn4czQ25IzznvTsnVLIHG/dHKHxSdJopcjvJD3FGEFCpkDvGE9XfmYpcuGPoxSQBlUq5g2glRAkn3EG0qu5c6Mhduq3KG2+GO4hOnyqg2nJH0aCZKzIJdxSNG1lCnzuKxpQsodTNU5stxGIb7igaF7KER+4oGm9kCU/cUTSOSGhszB1F4/V76YYsodSRZkmW8I07ikaPLKHUg7aQLOGUO4rGmSzhjTuKxpYsodjbYGQJuYNoUS29pU4WNDveGakDTaczIEoo+CSYZt0mdVWaoemmcjspVTeVutP2F8XCTe5+cIbiIQbcIZ5zf4hSV905933vAXeEMq7Dqegz7rtPp4CS58Kc2xuG8GHmzuVFWOqBxQP7m21r7qabst2Sknpc8ZtntwI/Sb5I8yA+WQRctShgx2Z3+Ohxt7miqpeHLm0LWPXCfmuu6f9U5eKC1KsJJVLTWeMifrWtlRh9oCd1h9uINyrNN2zXJFHg6ZfA10X7htACs15xZ91FYs8nKusHi8dRZzkKWt89H3nBLfQz4S14ib4JAAAAAAAAAC8n7qfpdjYJ/dGdH05m2zTtt7ra3rdtkIyi5al4r+06XkbTW7Bt66ZNkHxszAoQrDbdRPCFyyLxeVr9iHQ8TNqxvZgmXZsD4O+U3ZvwQ5pg4f4N4mYqtsfOorJapaaua3mVhuIz3SdBd+uzpLlkHhEU3ikIOZMxj2wtxk1T4yH/jegb3WeVxT5Dzgc5sC2QWM0H14MMaOrtmDhwTCAB3df3JjZNZ2w4X+MZU+rJz0zU2IrOLy6T34BmKkYP6p4fnjk2MKzSfc5sp/YqWTy/wJ/qvQse05VNsPdW44tySlbMy8m1tjE1pXr/c7WrKSJJYUsaNV16b34Zo1fLlxnlN0WbVEMtXrqSEDToV6n1vcjbWVEHpCrcSYe6Uq2ccTS3pw3oWmm9DrR7qvzL0d9IF6ixvE5KPO1TFS2hRfmuSFcBihLlaCprPZOjXNdQFXmmRflFX5c7TCEkREIk5IeESIiE/JAQCZGQHxIiIRLyQ8IqJG6XKtUjTBj2uvL0JP/HHgAAAAAAAAAAAAAAAAAAAPyf/gCZuUm9oVCsnAAAAABJRU5ErkJggg==",
      },
      { text: datajson.message, side: MessageSide.Right, icon: crow },
    ]);
  };

  const onInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setCurrentMessage(event.target.value);
  };

  const onServerChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setServer(event.target.value);
  };

  const onSend = () => {
    setMessages([
      ...messages,
      {
        text: currentMessage,
        icon: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAOEAAADhCAMAAAAJbSJIAAAAeFBMVEX///8AAACAgIB8fHz19fXp6emNjY37+/umpqbY2NiYmJiwsLB2dnb4+PgwMDCtra2UlJTKysq/v7/R0dGdnZ1eXl4qKipvb2/v7+9ISEhRUVHg4OATExMLCwsiIiJnZ2eIiIhBQUE6OjrDw8NFRUVZWVkbGxu4uLjqLqXEAAAEJElEQVR4nO3di3aiMBAGYOMNEPCCN6pWq1Z9/zfcspazPZZISAZmcP/vCfIfYhKSMHY6AAAAAAAAAAAAAAAAAAAAAK+nPwhCf5jxJ8Ggz90cYvF8sVypn1aX0fx1Us6jnSryvp5xN42CNylMl/Nj7gY68vz904BK7Ratzjgfl+TL7M/czbTmrQ3yZQ4tHXOCq2HAL3Puxtp4PsI8WnA3t7qPSgGVirgbXJXpT/CfJXeTqzlUDtiyiJFFwK8hlbvZ5oZWAZXqcjfclG8ZUKkRd9PNBNYBWzIveiYrNZ19G1Y3dqNMrgWjzdwpoFLyl+EufTSz4w5Qxn4czQ25IzznvTsnVLIHG/dHKHxSdJopcjvJD3FGEFCpkDvGE9XfmYpcuGPoxSQBlUq5g2glRAkn3EG0qu5c6Mhduq3KG2+GO4hOnyqg2nJH0aCZKzIJdxSNG1lCnzuKxpQsodTNU5stxGIb7igaF7KER+4oGm9kCU/cUTSOSGhszB1F4/V76YYsodSRZkmW8I07ikaPLKHUg7aQLOGUO4rGmSzhjTuKxpYsodjbYGQJuYNoUS29pU4WNDveGakDTaczIEoo+CSYZt0mdVWaoemmcjspVTeVutP2F8XCTe5+cIbiIQbcIZ5zf4hSV905933vAXeEMq7Dqegz7rtPp4CS58Kc2xuG8GHmzuVFWOqBxQP7m21r7qabst2Sknpc8ZtntwI/Sb5I8yA+WQRctShgx2Z3+Ohxt7miqpeHLm0LWPXCfmuu6f9U5eKC1KsJJVLTWeMifrWtlRh9oCd1h9uINyrNN2zXJFHg6ZfA10X7htACs15xZ91FYs8nKusHi8dRZzkKWt89H3nBLfQz4S14ib4JAAAAAAAAAC8n7qfpdjYJ/dGdH05m2zTtt7ra3rdtkIyi5al4r+06XkbTW7Bt66ZNkHxszAoQrDbdRPCFyyLxeVr9iHQ8TNqxvZgmXZsD4O+U3ZvwQ5pg4f4N4mYqtsfOorJapaaua3mVhuIz3SdBd+uzpLlkHhEU3ikIOZMxj2wtxk1T4yH/jegb3WeVxT5Dzgc5sC2QWM0H14MMaOrtmDhwTCAB3df3JjZNZ2w4X+MZU+rJz0zU2IrOLy6T34BmKkYP6p4fnjk2MKzSfc5sp/YqWTy/wJ/qvQse05VNsPdW44tySlbMy8m1tjE1pXr/c7WrKSJJYUsaNV16b34Zo1fLlxnlN0WbVEMtXrqSEDToV6n1vcjbWVEHpCrcSYe6Uq2ccTS3pw3oWmm9DrR7qvzL0d9IF6ixvE5KPO1TFS2hRfmuSFcBihLlaCprPZOjXNdQFXmmRflFX5c7TCEkREIk5IeESIiE/JAQCZGQHxIiIRLyQ8IqJG6XKtUjTBj2uvL0JP/HHgAAAAAAAAAAAAAAAAAAAPyf/gCZuUm9oVCsnAAAAABJRU5ErkJggg==",
      },
    ]);
    getMessage(currentMessage, state);
    setCurrentMessage("");
  };

  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <div className="App">
        <Avatar
          src={crow}
          sx={{
            width: 64,
            height: 64,
            position: "absolute",
            left: 32,
            top: 32,
            filter: "invert(1)",
          }}
        />
        <h1>Chat-RWKV</h1>
        <div className="message-box">
          {messages
            .map((message, index) => (
              <Message
                key={index}
                {...{ ...message, index: messages.length - index }}
              />
            ))
            .reverse()}
        </div>

        <TextField
          id="serverSet"
          label="Server"
          variant="outlined"
          value={server}
          onChange={onServerChange}
          color="primary"
          sx={{
            position: "absolute",
            right: 32,
            top: 32,
          }}
        />

        <TextField
          id="outlined-basic"
          multiline
          label="Message"
          variant="outlined"
          fullWidth
          value={currentMessage}
          onChange={onInputChange}
          color="primary"
          InputProps={{
            endAdornment: (
              <Button variant="contained" onClick={onSend}>
                Send
              </Button>
            ),
          }}
        />
      </div>
    </ThemeProvider>
  );
}

export default App;
