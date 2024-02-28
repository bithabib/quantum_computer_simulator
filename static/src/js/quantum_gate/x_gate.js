// include another js file
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

function moveToGate() {
  const moving_qbit = document.getElementById("drop_place");
  const drop_place1 = document.getElementById("drop_place1");
  const drop_place2 = document.getElementById("drop_place2");
  const drop_place3 = document.getElementById("drop_place3");
  const drop_place4 = document.getElementById("drop_place4");

  const distance1 = drop_place1.offsetLeft - moving_qbit.offsetLeft;
  const distance2 = drop_place2.offsetLeft - drop_place1.offsetLeft;
  const distance3 = drop_place3.offsetLeft - drop_place2.offsetLeft;
  const distance4 = drop_place4.offsetLeft - drop_place3.offsetLeft;

  const step1 = distance1 / 100;
  const step2 = distance2 / 100;
  const step3 = distance3 / 100;
  const step4 = distance4 / 100;

  currentPosition = moving_qbit.offsetLeft;
  const targets = [
    { dropPlace: drop_place1, step: step1 },
    { dropPlace: drop_place2, step: step2 },
    { dropPlace: drop_place3, step: step3 },
    { dropPlace: drop_place4, step: step4 },
  ];

  // Function to move the element to the next target position
  function moveToNextTarget(index) {
    if (index >= targets.length) {
      // All targets reached, animation finished
      return;
    }
    const target = targets[index];
    const interval = setInterval(() => {
      currentPosition += target.step;

      if (currentPosition >= target.dropPlace.offsetLeft) {
        if (index === 3) {
          let i = 1;
          while (i <= 35) {
            currentPosition += target.step;
            moving_qbit.style.left = currentPosition + "px";
            i++;
          }
        }
        clearInterval(interval); // Stop the animation when the target position is reached
        // Move to the next target
        // get the child of the drop_place
        var child = target.dropPlace.children[0];
        if (child) {
          // read the text of the child
          var text = child.textContent;
          console.log(text);
          if (text === "X") {
            rotate_state("x", math.PI);
          } else if (text === "Y") {
            rotate_state("y", math.PI);
          } else if (text === "Z") {
            rotate_state("z", math.PI);
          } else if (text === "H") {
            rotate_state("y", math.PI / 2);
            rotate_state("x", math.PI);
          }
          // rotate_state("x", math.PI);
        }
        setTimeout(() => {
          moveToNextTarget(index + 1); // Move to the next target after the pause
        }, 1000);
      } else {
        moving_qbit.style.left = currentPosition + "px";
      }
    }, 10);
  }
  moveToNextTarget(0);
}
