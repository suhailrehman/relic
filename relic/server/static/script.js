$("form#relicform").submit(function(e){

    e.preventDefault();

    // Hack to send unchecked boxes in POST
    $(".form-check-input").each(function(index) {
        console.log(index + ": " + $(this).attr('id') + " Checked?: " + $(this).is(':checked'));
        if ($(this).is(':checked')) {
            document.getElementById($(this).attr('id') + 'Hidden').disabled = true;
            console.log(document.getElementById($(this).attr('id') + 'Hidden'))
        }
    });

    var formData = new FormData($(this)[0]);
    console.log(formData)

    $.ajax({
        url: 'upload',
        type: 'POST',
        data: formData,
        async: false,
        success: function (data) {
            console.log(data)
            $("#mynetwork").html("Graph will be displayed once the Job is finished...")
            $("#artifact").html("")
            $("#artifact_header").attr('hidden', true)
            updateJobStatus(data['job_id'])
        },
        cache: false,
        contentType: false,
        processData: false
    });

    return false;
});


function renderGraph(jobId){
    var width = $("#mynetwork").width();
    var height = $("#mynetwork").height();
    $("#mynetwork").html("Loading Canned Demo...")
    $.ajax({
        url: 'render/' + jobId,
        async: false,
        cache: false,
        data: 'width='+width+'&height='+height,
        success: function (data) {
            //console.log(data)
            $("#mynetwork").html(data)
            $(".graph-button").attr('hidden', false)
            $("#export").attr('download', jobId+'.csv').attr('onclick', "location.href='export/"+jobId+"'");

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
                        $("#artifact_header").html('Artifact Inspector: '+artifact_id).removeAttr('hidden');
                        $($.fn.dataTable.tables(true)).DataTable().columns.adjust();

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

});


