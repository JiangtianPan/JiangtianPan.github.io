(function($) {

	"use strict";

	$('#sidebarCollapse').on('click', function () {
      $('#sidebar').toggleClass('active');
  });

})(jQuery);
var clicks = 0;
function openNav() {
	clicks += 1;
	if(clicks % 2 == 0){
		document.getElementById("content").style.marginLeft = "250px";
	} else {
		document.getElementById("content").style.marginLeft= "0";
	}
}


