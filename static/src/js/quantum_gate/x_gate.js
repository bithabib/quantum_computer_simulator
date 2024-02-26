function allowDrop(ev) {
  ev.preventDefault();
}
function drag(ev) {
  ev.dataTransfer.setData("text", ev.target.id);
  // get the parent of the ev
  var parent = ev.target.parentNode;
  // get the id of the paren of the ev
  var parent_id = parent.id;
  // check if the parent id contain drop_place
  if (parent_id.includes("drop_place")) {
    parent.style.height = "20px";
    parent.style.width = "20px";
  }
}
function drop(ev) {
  ev.preventDefault();
  var data = ev.dataTransfer.getData("text");
  // change the style of the ev
  ev.target.style.height = "60px";
  ev.target.style.width = "60px";
  ev.target.appendChild(document.getElementById(data));
}
