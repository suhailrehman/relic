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

        xhr: function() {
            var xhr = new window.XMLHttpRequest();

            // First turn the Infer Lineage Button into a progress bar


            // Upload progress
            xhr.upload.addEventListener("progress", function(evt){
                if (evt.lengthComputable) {
                    var percentComplete = (evt.loaded / evt.total) * 100;
                    //Do something with upload progress
                    $('#formsubmit').replaceWith('<div id="formsubmit" class="progress"><div className="progress-bar progress-bar-striped progress-bar-animated"  aria-valuenow="'+percentComplete+'" style="width:'+percentComplete+'%">Uploading...</div></div>');
                    console.log('Upload%: '+percentComplete);
                }
           }, false);

           return xhr;
       },

        success: function (data) {
            console.log(data)
            $("#mynetwork").html("Graph will be displayed once the Job is finished...")
            $("#artifact").html("")
            $("#artifact_header").attr('hidden', true)
            updateJobStatus(data['job_id'])
        },

        complete: function(data){
            $('#formsubmit').replaceWith('<button class="btn btn-success" type="submit" id="formsubmit">Infer Lineage</button>');
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
    var jobType = encodeURIComponent($("#jobtype").val());
    var renderData = 'width='+width+'&height='+height;
    if(jobType != 'null')
        renderData += '&job_type='+jobType;
    $("#mynetwork").html("Loading...")
    $.ajax({
        url: 'render/' + encodeURIComponent(jobId),
        cache: false,
        data: renderData,
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

$("button#jobsubmit").click(function(e){
    e.preventDefault();

    console.log('Clicked Job Request')
    console.log($("#jobid").val())
    console.log($("#jobtype").val())
    renderGraph($("#jobid").val())


});

function updateJobStatus(jobId){

    var steps=1000

    $.ajax({
        url: 'status/'+jobId,
        dataType: "json",
        cache: false,
        success: function(data){
            console.log(data)

            if(data['status'] === 'running')
            {
                var pcg = Math.floor(data['phaseno']/data['totalphases']*100);
                $("h3#status").html("Running Relic Job Id: "+jobId)
                $("h4#running").html("Currently Running: "+ data['current_phase'])
                $('div#pbar').attr('aria-valuenow',pcg).attr('style','width:'+Number(pcg)+'%')
                $("div#alert_container").removeClass("collapse alert-success").addClass("alert-primary");
            }
            else if(data['status'] === 'pending')
            {
                var pcg = 0;
                $("h3#status").html("Preparing Relic Job Id: "+jobId)
                $('div#pbar').attr('aria-valuenow',pcg).attr('style','width:'+Number(pcg)+'%')
                $("div#alert_container").removeClass("collapse alert-success").addClass("alert-primary");
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
    // var myModal = document.getElementById('myModal')
    // var myInput = document.getElementById('myInput')
    //
    // myModal.addEventListener('shown.bs.modal', function () {
    //   myInput.focus()
    // })
});


