const fs = require ("fs")  
var decoder = fs.readFileSync("./20B_tokenizer.json")
decoder = JSON.parse(decoder)

    fs.writeFileSync("./decoder.json", JSON.stringify(Object.entries(decoder.model.vocab).reduce(
      (acc, n) => ({ ...acc, [n[1]]: n[0] }),
      {}
    ))
    )
