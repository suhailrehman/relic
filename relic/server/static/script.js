$("form#relicform").submit(function(e){

    e.preventDefault()
    var formData = new FormData($(this)[0]);
    console.log(formData)

    $.ajax({
        url: 'upload',
        type: 'POST',
        data: formData,
        async: false,
        success: function (data) {
            console.log(data)
            updateJobStatus(data['job_id'])
        },
        cache: false,
        contentType: false,
        processData: false
    });

    return false;
});


function renderGraph(jobId){
    $.ajax({
        url: 'render/' + jobId,
        async: false,
        cache: false,
        success: function (data) {
            //console.log(data)
            $("div#mynetwork").html(data)

            network.on('click', function (properties) {
            var nodeID = properties.nodes[0];
            if (nodeID) {
                var clickedNode = this.body.nodes[nodeID];
                console.log('clicked node:', clickedNode.options.label);
                //console.log('pointer', properties.pointer);
                var artifact_id = clickedNode.options.label;

                $.ajax({
                    url: 'artifact/' + jobId + '/' + artifact_id,
                    async: false,
                    cache: false,
                    success: function (data) {
                        //console.log(data)
                        $("div#artifact").html(data)
                        $('#artifacttable').DataTable();
                        $('.dataTables_length').addClass('bs-select');
                        $( "#artifact_header").removeAttr('hidden');

                    }
                });

            }
        });

        }
    });
}

function updateJobStatus(jobId){

    var steps=1000

    $.ajax({
        url: 'status/'+jobId,
        dataType: "json",
        async: false,
        cache: false,
        success: function(data){
            console.log(data)

            if(data['status'] === 'running')
            {
                var pcg = Math.floor(data['phaseno']/data['totalphases']*100);
                $("h3#status").html("Running Relic Job Id: "+jobId)
                $("h4#running").html("Currently Running: "+ data['current_phase'])
                $('div#pbar').attr('aria-valuenow',pcg).attr('style','width:'+Number(pcg)+'%')
                $("div#alert_container").removeClass("collapse").addClass("alert-primary");
            }
            else if(data['status'] === 'pending')
            {
                var pcg = 0;
                $("h3#status").html("Preparing Relic Job Id: "+jobId)
                $('div#pbar').attr('aria-valuenow',pcg).attr('style','width:'+Number(pcg)+'%')
                $("div#alert_container").removeClass("collapse").addClass("alert-primary");
            }
            else if (data['status'] === 'complete')
            {
                $("h3#status").html("Relic Job Id: "+jobId+" Completed");
                $("h4#running").html("");
                $("div#alert_container").removeClass('alert-primary').addClass("alert-success");
                $('div#pbar').attr('aria-valuenow',100).attr('style','width:'+Number(100)+'%').addClass("bg-success");
            }

            steps -= 1
        },
        complete: function(jqXHR, status){
            data = $.parseJSON(jqXHR.responseText)
            if(data['status'] !== 'complete' && steps > 0)
            {
                setTimeout(updateJobStatus.bind(null, jobId), 500);
            }
            else
            {
                renderGraph(jobId)
            }

        }
    });
}


$(document).ready(function () {
                    $('#dtBasicExample').DataTable();
                    $('.dataTables_length').addClass('bs-select');
});


