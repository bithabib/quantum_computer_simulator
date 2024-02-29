// include another js file
function allowDrop(ev) {
  ev.preventDefault();
}
function drag(ev) {
  ev.dataTransfer.setData("text", ev.target.id);
}
function drop(ev) {
  ev.preventDefault();
  var data = ev.dataTransfer.getData("text");
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
          while (i <= 100) {
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

function selectedQbit(event) {
  const selected_qbit = event.target;
  if (selected_qbit.id === "checkbox_single_qbit") {
    // lets see its checked or not
    if (selected_qbit.checked) {
    } else {
    }
  } else {
    // lets see its checked or not
    const sm_qbit_gate_holder = document.getElementById("sm_qbit_gate_holder");
    if (selected_qbit.checked) {
      document.getElementById("drop_place_multi").style.visibility = "visible";
      sm_qbit_gate_holder.classList.add("checked");
    } else {
      document.getElementById("drop_place_multi").style.visibility = "hidden";
      sm_qbit_gate_holder.classList.remove("checked");
    }
  }
}
