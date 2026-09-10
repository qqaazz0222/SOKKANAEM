-- Make the standalone TeX compile inside its own delivery directory.
function Image(el)
  el.src = el.src:gsub('^streaming_draft/', '')
  return el
end

function Table(el)
  local widths = {
    [5] = {0.19, 0.28, 0.25, 0.10, 0.18},
    [6] = {0.20, 0.16, 0.16, 0.16, 0.18, 0.14},
    [7] = {0.10, 0.12, 0.12, 0.12, 0.16, 0.18, 0.20}
  }
  local w = widths[#el.colspecs]
  if w then
    for i, spec in ipairs(el.colspecs) do
      el.colspecs[i] = {spec[1], w[i]}
    end
  end
  return el
end

-- Keep the full source caption with its image, without duplicate numbering.
function Pandoc(doc)
  local out = pandoc.List()
  local i = 1
  while i <= #doc.blocks do
    local b = doc.blocks[i]
    local n = doc.blocks[i+1]
    if b.t == 'Figure' and n and n.t == 'Para' and pandoc.utils.stringify(n):match('^Figure %d+%.') then
      local plain = pandoc.utils.stringify(n):gsub('^Figure %d+%.%s*', '')
      b.caption.long = pandoc.Blocks({pandoc.Para({pandoc.Str(plain)})})
      b.caption.short = nil
      out:insert(b)
      i = i + 2
    else
      out:insert(b)
      i = i + 1
    end
  end
  doc.blocks = out
  return doc
end
